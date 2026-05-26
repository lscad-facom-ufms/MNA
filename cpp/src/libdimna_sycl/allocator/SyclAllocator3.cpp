module;
#include <stdexcept>
#include <chrono>
#include <tabulate/table.hpp>
#include <sycl/sycl.hpp>
module dimna.allocator.sycl_allocator3;

import mna.model.job;
import mna.model.soa;

import dimna.model.solution;
import dimna.model.times;
import dimna.allocator.constants;

import dimna.policy.latency.default_latency;

namespace mna::di::par {

Result 
SyclAllocator3::run_mna_jobs(JobVector& jobs){
  
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

  int kar = 0;

  // Loop principal de soluções das jobs;
  for (auto& job : jobs) {

    // std::cout << "Job " << kar++ << "\n";
    
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
    
    sycl::buffer<int, 1> buf_vec_L{filtered_latencies};

    // std::cout << "Filtrou em "<< diff_in_seconds.count() << " s\n";
    // Operador e identidade da redução
    using minSolution = sycl::minimum<mna::di::SolutionTriplet>;
    mna::di::SolutionTriplet identity = {INF32, -1, -1, -1};
    mna::di::SolutionTriplet job_solution = identity;
    sycl::buffer<mna::di::SolutionTriplet, 1> buf_job_solution{&job_solution, sycl::range<1>{1}};

    // std::cout << "f_num_nodes: " << f_num_nodes << ", size: " << latencies.size() << "\n";

    const auto local_num_nodes = f_num_nodes;
    start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler& h) {

      sycl::accessor ac_vec_R{buf_vec_R, h, sycl::read_only};
      sycl::accessor ac_vec_B{buf_vec_B, h, sycl::read_only};
      sycl::accessor ac_vec_L{buf_vec_L, h, sycl::read_only};
      // sycl::accessor ac_vec_busy{buf_vec_busy, h, sycl::read_only};

      auto red = sycl::reduction(buf_job_solution, h, identity, minSolution());
        
      h.parallel_for(sycl::range<1>{local_num_nodes}, red, [=](sycl::id<1> idx, auto& res_reducer) {

        int i = idx[0];

        int r_i = ac_vec_R[i];
        int b_i = ac_vec_B[i];
        int l_i = l_job - ac_vec_L[i];

        // Campeão local desta thread
        mna::di::SolutionTriplet best_local = identity;
        
        // Loop ainda desbalanceado
        for (int j = i; j < local_num_nodes; ++j) {

          int is_j_diff = (i != j) ? 1 : 0;

          int partial_l_ij = l_i - (ac_vec_L[j] * is_j_diff);
          
          if (partial_l_ij < 0)
            continue;

          int partial_r_ij = r_i + (ac_vec_R[j] * is_j_diff);
          int partial_b_ij = b_i + (ac_vec_B[j] * is_j_diff);

          for (int k = j; k < local_num_nodes; ++k){

            int is_k_diff = (j != k) ? 1 : 0;

            int val_l = partial_l_ij - (ac_vec_L[k] * is_k_diff);

            if (val_l < 0) 
              continue;

            int total_r = partial_r_ij + (ac_vec_R[k] * is_k_diff) - job_r;
            int total_b = partial_b_ij + (ac_vec_B[k] * is_k_diff) - job_b;

            // Verifica se atende aos recursos
            if (total_r >= 0 && total_b >= 0) {
                    
              int val_f = (total_r * total_r) + (total_b * total_b) - val_l;

              if (val_f < best_local.of) {
                best_local = {val_f, i, j, k};
              }
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

      auto [of, f_node_i, f_node_j, f_node_k] = ac_job_solution[0];

      if (of < INF32) {
        
        auto node_i = mapping[f_node_i];
        auto node_j = mapping[f_node_j];
        auto node_k = mapping[f_node_k];

        // std::cout << "Of:" << of << " i:" << node_i << " j:" << node_j << " k:" << node_k << "\n";

        nodes.busy[node_i] = 1;
        
        nodes.busy[node_j] = 1;

        nodes.busy[node_k] = 1;
        
        std::vector<int> selected_nodes;
        selected_nodes.push_back(node_i);

        if (node_i != node_j)
          selected_nodes.push_back(node_j);

        if (node_k != node_i && node_k != node_j) 
          selected_nodes.push_back(node_k);

        solutions.push_back({of, selected_nodes});
      }
      else{
        // Sets OF to zero if no valid combination was found
        solutions.push_back({0, std::vector<int>()});
      }
    }
    q.wait();

  }
  q.wait();

  return Result{solutions, times};
}

} // namespace mna::di::par

