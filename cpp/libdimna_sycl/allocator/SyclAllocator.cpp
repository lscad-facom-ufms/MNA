module;
#include <stdexcept>
#include <chrono>
#include <tabulate/table.hpp>
#include <sycl/sycl.hpp>
module dimna.allocator.sycl_allocator;

import mna.model.job;
import mna.model.soa;

import dimna.model.solution;
import dimna.model.times;
import dimna.allocator.constants;

import dimna.policy.latency.default_latency;

namespace mna::di::par {

Result 
SyclAllocator::run_mna_jobs(JobVector& jobs){
  
  // Declaração da fila de execução na GPU
  sycl::queue q{sycl::gpu_selector_v};

  // Variável para cálculo de tempo de cada etapa
  Times times{};

  // Vector que contém as soluções para cada job
  std::vector<Solution> solutions;
  
  // Vector que guarda as latências dos caminhos da rede, dado o nó de entrada da job
  std::vector<int32_t> latencies(num_nodes);

  // Buffers para acesso dos parâmteros dos nós na GPU
  auto& nodes = network->vertexes();

  // sycl::buffer<int, 1> buf_vec_R{nodes.resource};
  // sycl::buffer<int, 1> buf_vec_B{nodes.bandwidth};
  // sycl::buffer<int, 1> buf_vec_busy{nodes.busy};

  int k = 0;

  // Loop principal de soluções das jobs;
  for (auto& job : jobs) {

    // std::cout << "Job " << k++ << "\n";
    
    // Captura do tempo da iteração principal
    auto start = std::chrono::steady_clock::now();

    auto job_r = job.resource;
    auto job_b = job.bandwidth;
    auto l_job = job.latency - t_c;

    auto origin = job.origin;

    lp.get_latencies(network, origin, latencies);
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff_in_seconds = end - start;

    times.latencies += diff_in_seconds.count();
    // std::cout << "Calculou latências em "<< diff_in_seconds.count() << " s\n";

    sycl::buffer<int, 1> buf_vec_L{latencies.data(), sycl::range<1>{latencies.size()}};

    start = std::chrono::steady_clock::now();

    auto [filtered_nodes, filtered_latencies, mapping, f_num_nodes] = this->filter_nodes(job, latencies);

    end = std::chrono::steady_clock::now();
    diff_in_seconds = end - start;
    times.filter += diff_in_seconds.count();

    if (f_num_nodes == 0) {
      solutions.push_back({0, std::vector<int>()});
      continue;
    }

    sycl::buffer<int, 1> buf_vec_R{filtered_nodes.resource};
    sycl::buffer<int, 1> buf_vec_B{filtered_nodes.bandwidth};
    
    // std::cout << "Filtrou em "<< diff_in_seconds.count() << " s\n";
    // Operador e identidade da redução
    using minSolution = sycl::minimum<mna::di::SolutionPair>;
    mna::di::SolutionPair identity = {INF32, -1, -1};
    mna::di::SolutionPair job_solution = identity;
    sycl::buffer<mna::di::SolutionPair, 1> buf_job_solution{&job_solution, sycl::range<1>{1}};

    // std::cout << "num_nodes: " << f_num_nodes << ", size: " << latencies.size() << "\n";

    const auto local_num_nodes = f_num_nodes;
    start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler& h) {

        sycl::accessor ac_vec_R{buf_vec_R, h, sycl::read_only};
        sycl::accessor ac_vec_B{buf_vec_B, h, sycl::read_only};
        sycl::accessor ac_vec_L{buf_vec_L, h, sycl::read_only};
        // sycl::accessor ac_vec_busy{buf_vec_busy, h, sycl::read_only};

        auto red = sycl::reduction(buf_job_solution, h, identity, minSolution());

        // O range volta a ser apenas num_nodes (100.000 threads)
        h.parallel_for(sycl::range<1>{num_nodes}, red, [=](sycl::id<1> idx, auto& res_reducer) {

            int i = idx[0];

            int r_i = ac_vec_R[i];
            int b_i = ac_vec_B[i];
            int l_i = ac_vec_L[i];

            if (l_i > l_job) {
                return;
            }
 
            // Campeão local desta thread
            mna::di::SolutionPair best_local = {INF32, -1, -1};
            
            // Loop ainda desbalanceado
            for (int j = i; j < local_num_nodes; ++j) {

                int l_j = ac_vec_L[j];
                if (l_j > l_job) {
                    continue;
                }

                int is_diff = (i != j) ? 1 : 0;

                int val_r = r_i + (ac_vec_R[j] * is_diff) - job_r;
                int val_b = b_i + (ac_vec_B[j] * is_diff) - job_b;
                int val_l = l_job - (l_i + l_j * is_diff);

                if (val_r >= 0 && val_b >= 0 && val_l >= 0) {
                    int val_f = (val_r * val_r) + (val_b * val_b) - val_l;

                    if (val_f < best_local.of) {
                        best_local = {val_f, i, j};
                    }
                }
            }

            // Combina apenas o melhor resultado encontrado pela thread
            if (best_local.of < INF32) {
                res_reducer.combine(best_local);
            }
        });
    });
    q.wait();

    end = std::chrono::steady_clock::now();
    diff_in_seconds = end - start;
    times.search += diff_in_seconds.count();
    // Finalização do processamento da job
    {
      sycl::host_accessor ac_job_solution{buf_job_solution, sycl::read_only};

      auto [of, f_node_i, f_node_j] = ac_job_solution[0];

      auto node_i = mapping[f_node_i];
      auto node_j = mapping[f_node_j];

      // std::cout << "Of:" << of << " i:" << node_i << " j:" << node_j << "\n";
      if (of < INF32) {
        
        nodes.busy[node_i] = 1;
        nodes.busy[node_j] = 1;
        
        std::vector<int> selected_nodes;
        selected_nodes.push_back(node_i);

        if (node_i != node_j)
          selected_nodes.push_back(node_j);
        
        solutions.push_back({of, selected_nodes});
      }
      else{
        // Sets OF to zero if no valid combination was found
        solutions.push_back({0, std::vector<int>()});
      }
    }
    q.wait();

    // std::cout << "Search em: "<< diff_in_seconds.count() << " s\n\n";

    // std::cout << "Fez tudo "<< diff_in_seconds.count() << " s\n\n";

    // if(k > 1){
      // std::cout << "---------------------\n";
      // return Result{solutions, times};
      // throw std::runtime_error("Acabou\n");
    // }
  }
  q.wait();

  return Result{solutions, times};
}

} // namespace mna::di::par
