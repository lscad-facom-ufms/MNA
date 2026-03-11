#include <iostream>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
import mna.io.cli_parser;
import mna.io.directory_scanner;
import mna.io.parquet_reader;
import mna.model.iot_network;
import mna.model.job;
import dimna.allocator.di_allocator;

int
main (int argc, char *argv[]) {
  mna::Config dir_config = mna::read_CLI(argc, argv);

  mna::InstancesMap instances = mna::scan_input_dir(dir_config.input_folder);
  
  mna::InstanceStructure lastInstance;

  for (const auto& [key, value] : instances)
    lastInstance = value;
  
  mna::ParquetReader pr;

  int total_vertexes = *(pr.get_num_rows(lastInstance.nodes));
  int total_edges = *(pr.get_num_rows(lastInstance.edges[0]));

  auto network = std::make_shared<mna::IoTNetwork>(total_vertexes);

  using vertex_type = decltype(network)::element_type::vertex_type;
  using edge_type = decltype(network)::element_type::edge_type;

  auto add_edge = [&](int32_t source, int32_t target, int32_t latency){
    network->add_edge(edge_type{target, latency}, source); 
  };
  auto add_vertex = [&](int64_t R, int64_t B, int64_t busy, int64_t inactive){
    network->add_vertex(vertex_type{R, B, busy, inactive});
  };

  mna::JobVector jobsV;

  auto add_job = [&](int64_t jr, int64_t jb, int64_t jl, int64_t jo){
    std::cout << jr << " " << jb << " " << jl << " " << jo << "\n";
    jobsV.push_back(mna::Job{jr, jb, jl, jo});
  };

  pr.read_vertexes(lastInstance.nodes, add_vertex);
  pr.read_edges(lastInstance.edges, add_edge);
  pr.read_jobs(lastInstance.jobs[0], add_job);

  int cut_comb_nodes = 8;
  int cut_sol = 2;

  mna::di::DiAllocator allocator(network, 2, cut_comb_nodes);
  
  std::vector<int32_t> latencies(network->vertex_count());
  allocator.get_latencies(jobsV[0].origin, latencies);

  int validation_l[] = {65, 25, 74, 31, 26, 77, 34, 48, 0, 21};

  for (int i = 0; i < latencies.size(); ++i){
    if(latencies[i] != validation_l[i])
      throw std::runtime_error("Error: Mismatch on obtained latencies");
  }

  std::set<int> validation_n{0, 1, 2, 4, 5, 7, 8, 9};
  
  std::set<int> valid_nodes = allocator.preselect_nodes(jobsV[0], latencies);

  for (auto i : validation_n){
    if(!valid_nodes.contains(i))
      throw std::runtime_error("Error: Valid node not include in the set");
  };

  auto [of, solution] = allocator.find_best_comb(jobsV[0], valid_nodes, latencies);
  
  for (auto i : solution)
    std::cout<< i << ", ";
  std::cout << "] "<< of << "\n";
  // std::cout << "\n" << jobsV.size() << "\n";
  return 0;
}
