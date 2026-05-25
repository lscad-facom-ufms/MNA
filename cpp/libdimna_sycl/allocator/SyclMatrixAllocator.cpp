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

  // Declaração dos buffers das matrizes que guardam os componentes do OF;
  sycl::buffer<int, 2> buf_mat_R{sycl::range<2>{num_nodes, num_nodes}};
  sycl::buffer<int, 2> buf_mat_B{sycl::range<2>{num_nodes, num_nodes}};
  sycl::buffer<int, 2> buf_mat_L{sycl::range<2>{num_nodes, num_nodes}};
  
  // Declaração do buffer da matriz resultante F
  sycl::buffer<int, 2> buf_mat_F{sycl::range<2>{num_nodes, num_nodes}};
  
  // Variável para cálculo de tempo de cada etapa
  Times times{};

  auto start = std::chrono::steady_clock::now();

  // Inicialização das matrizes R e B:
  {
    auto& nodes = network->vertexes();

    sycl::buffer buf_vec_R{nodes.resource};
    sycl::buffer buf_vec_B{nodes.bandwidth};

    auto ev_R = q.submit([&](sycl::handler &h){

      sycl::accessor ac_mat_R{buf_mat_R, h, sycl::write_only, sycl::no_init};

      sycl::accessor ac_vec_R{buf_vec_R, h, sycl::read_only};

      h.parallel_for(sycl::range{num_nodes, num_nodes}, [=](sycl::id<2> idx){
        int j = idx[0];
        int i = idx[1];

        // Faz a soma da combinação dos R dos índices, ignorando o segundo se i == j;
      ac_mat_R[i][j] = ac_vec_R[i] + ac_vec_R[j] * (i != j);

      });

    });

    auto ev_B = q.submit([&](sycl::handler &h){

      sycl::accessor ac_mat_B{buf_mat_B, h, sycl::write_only, sycl::no_init};

      sycl::accessor ac_vec_B{buf_vec_B, h, sycl::read_only};

      h.parallel_for(sycl::range{num_nodes, num_nodes}, [=](sycl::id<2> idx){
        int j = idx[0];
        int i = idx[1];

        // Faz a soma da combinação dos B dos índices, ignorando o segundo se i == j;
        ac_mat_B[i][j] = ac_vec_B[i] + ac_vec_B[j] * (i != j);
      });

    });

  }
  // Sincronização após inicialização das matrizes R e B
  q.wait();

  // Tempo do pré-processamento
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff_in_seconds = end - start;
  times.latencies += diff_in_seconds.count();
  
  std::cout << "Pré-processou em " << diff_in_seconds.count() << " s\n";

  // Vector que contém as soluções para cada job
  std::vector<Solution> solutions;
  
  // Vector que guarda as latências dos caminhos da rede, dado o nó de entrada da job
  std::vector<int32_t> latencies(num_nodes);


  // Buffer para acesso dos busy nodes na GPU
  auto& nodes = network->vertexes();
  sycl::buffer<int, 1> buf_vec_busy{nodes.busy};

  // Valores de job_r e job_b da última job processada, para serem usados na atualização de R e B;
  int last_r = 0, last_b = 0;

  int k = 0;

  // Loop principal de soluções das jobs;
  for (auto& job : jobs) {

    std::cout << "Job " << k++ << "\n";
    
    // Captura do tempo da iteração principal
    start = std::chrono::steady_clock::now();


    auto start_real = std::chrono::steady_clock::now();
    
    // Atualização da matriz R para a job atual
    auto job_r = job.resource;

    
    auto ev_R = q.submit([&](sycl::handler &h){

      sycl::accessor ac_mat_R{buf_mat_R, h, sycl::read_write};

      h.parallel_for(sycl::range{num_nodes, num_nodes}, [=](sycl::id<2> idx){
        int j = idx[0];
        int i = idx[1];

        // Retorna o valor de R para o original e aplica o valor job_r atual
        ac_mat_R[i][j] = ac_mat_R[i][j] + last_r - job_r;

      });

    });
    
    // Atualização da matriz B para a job atual
    auto job_b = job.bandwidth;

    auto ev_B = q.submit([&](sycl::handler &h){

      sycl::accessor ac_mat_B{buf_mat_B, h, sycl::read_write};

      h.parallel_for(sycl::range{num_nodes, num_nodes}, [=](sycl::id<2> idx){
        int j = idx[0];
        int i = idx[1];

        // Retorna o valor de B para o original e aplica o valor job_b atual
        ac_mat_B[i][j] = ac_mat_B[i][j] + last_b - job_b;

      });

    });

    // Chamada da função get_latencies() fora do grafo de tasks :(

    auto origin = job.origin;

    lp.get_latencies(network, origin, latencies);

    // Buffer para acesso das latências na GPU
    sycl::buffer<int, 1> buf_vec_L{latencies};
    
    // Sincronização de todos os elementos antes do cálculo da matriz L
    q.wait();


    end = std::chrono::steady_clock::now();
    diff_in_seconds = end - start;
    times.latencies += diff_in_seconds.count();

    std::cout << "Calculou latências,  R e B em "<< diff_in_seconds.count() << " s\n";

    start = std::chrono::steady_clock::now();
    // Cálculo da matriz L 
    auto l_job = job.latency - t_c;

    auto ev_L = q.submit([&](sycl::handler &h){

      sycl::accessor ac_mat_L{buf_mat_L, h, sycl::write_only, sycl::no_init};

      sycl::accessor ac_vec_L{buf_vec_L, h, sycl::read_only};

      h.parallel_for(sycl::range{num_nodes, num_nodes}, [=](sycl::id<2> idx){
        int j = idx[0];
        int i = idx[1];

        // Faz a soma da combinação dos L dos índices, ignorando o segundo se i == j, e diminui l_job;
        ac_mat_L[i][j] = l_job - (ac_vec_L[i] + ac_vec_L[j] * (i != j));

      });

    });
    q.wait();

    end = std::chrono::steady_clock::now();
    diff_in_seconds = end - start;
    times.latencies += diff_in_seconds.count();

    std::cout << "Calculou L em "<< diff_in_seconds.count() << " s\n";

    start = std::chrono::steady_clock::now();

    auto ev_F = q.submit([&](sycl::handler &h){

      h.depends_on({ev_R, ev_B, ev_L});

      // Accessor para as matrizes que compoem o OF
      sycl::accessor ac_mat_R{buf_mat_R, h, sycl::read_only};
      sycl::accessor ac_mat_B{buf_mat_B, h, sycl::read_only};
      sycl::accessor ac_mat_L{buf_mat_L, h, sycl::read_only};
      sycl::accessor ac_vec_L{buf_vec_L, h, sycl::read_only};
      
      // Accessor para o vector de nós com a flag de disponibilidade
      sycl::accessor ac_vec_busy{buf_vec_busy, h, sycl::read_only};
      
      // Accessor para a matriz F com o OF resultante
      sycl::accessor ac_mat_F{buf_mat_F, h, sycl::write_only, sycl::no_init};

      h.parallel_for(sycl::range{num_nodes, num_nodes}, [=](sycl::id<2> idx){
        int j = idx[0];
        int i = idx[1];

        /* Calcula o OF da combinação, jogando pra infinito caso seja inválida,
         * isto é, caso um dos nós já esteja ocupado, ou se a combinação não 
         * atende aos requisitos de r, b ou latência da job.
        */
        
        if (ac_vec_busy[i] || ac_vec_busy[j]) 
          ac_mat_F[i][j] = INF32;

        else if (ac_vec_L[i] > l_job || ac_vec_L[j] > l_job || ac_mat_L[i][j] < 0)
          ac_mat_F[i][j] = INF32;

        else if (ac_mat_R[i][j] < 0 || ac_mat_B[i][j] < 0)
          ac_mat_F[i][j] = INF32;

        else{
          auto f0 = ac_mat_R[i][j];
          auto f1 = ac_mat_B[i][j];
          auto f2 = ac_mat_L[i][j];

          ac_mat_F[i][j] = (f0 * f0) + (f1 * f1) - f2;
        }
      });

    });

    q.wait();

    // Tempo levado para o cálculo das matrizes 
    end = std::chrono::steady_clock::now();
    diff_in_seconds = end - start;
    times.filter += diff_in_seconds.count();
    times.latencies += diff_in_seconds.count();

    std::cout << "Calculou F em "<< diff_in_seconds.count() << " s\n";

    // std::cout << "Gerou matriz F\n";
    
    // Operador utilizado na redução
    using minSolution = sycl::minimum<SolutionPair>;
    
    // Valor padrão que a redução usará internamente para a variável
    SolutionPair identity = {INF32, -1, -1};

    // Inicialização das variáveis que representam a solução
    SolutionPair job_solution = identity;

    // Buffer para a variável da solução para uma job, onde estará o resultado da redução
    sycl::buffer<SolutionPair, 1> buf_job_solution{&job_solution, sycl::range<1>{1}};
    
    // Captura do tempo para a redução final
    start = std::chrono::steady_clock::now();
    
    // Redução na matriz F para encontrar a melhor combinação de nós pelo OF
    auto ev_reduction = q.submit([&](sycl::handler& h){

      h.depends_on(ev_F);

      sycl::accessor ac_mat_F{buf_mat_F, h, sycl::read_only};
      
      // Variável que representa a operação de redução em si
      auto red = sycl::reduction(buf_job_solution, h, identity, minSolution());

      h.parallel_for(sycl::range<2>{num_nodes, num_nodes}, red, [=](sycl::id<2> idx, auto& res_reducer){
        
        int j = idx[0];
        int i = idx[1];
        
        mna::di::SolutionPair partial = {ac_mat_F[i][j], i, j};
            
        res_reducer.combine(partial);
      });
    });
    q.wait();

    // std::cout << "Reduziu\n\n";

    end = std::chrono::steady_clock::now();
    diff_in_seconds = end - start;
    times.search += diff_in_seconds.count();

    std::cout << "Fez a redução em "<< diff_in_seconds.count() << " s\n";
    // Finalização do processamento da job
    {
      sycl::host_accessor ac_job_solution{buf_job_solution, sycl::read_only};

      auto [of, i, j] = ac_job_solution[0];

      if (of < INF32){
        
        // Marcação dos nós selecionados como busy
        sycl::host_accessor ac_vec_busy{buf_vec_busy, sycl::read_write};

        ac_vec_busy[i] = 1;
        ac_vec_busy[j] = 1;

        // Salva a solução, eliminando nós duplicados
        std::vector<int> selected_nodes;
        selected_nodes.push_back(i);

        if (i != j)
          selected_nodes.push_back(j);
        
        solutions.push_back({of, selected_nodes});
      }
      else{
        // Sets OF to zero if no valid combination was found
        solutions.push_back({0, std::vector<int>()});
      }

      last_r = job_r;
      last_b = job_b;
    }
    q.wait();


    end = std::chrono::steady_clock::now();
    diff_in_seconds = end - start_real;
    times.search += diff_in_seconds.count();

    std::cout << "Fez tudo "<< diff_in_seconds.count() << " s\n\n";

    if(k > 100){
      std::cout << "---------------------\n";
      return Result{solutions, times};
      // throw std::runtime_error("Acabou\n");
    }
  }
  q.wait();

  return Result{solutions, times};
}

} // namespace mna::di::par
