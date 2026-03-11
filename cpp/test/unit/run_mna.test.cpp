#include <iostream>
#include <vector>
#include <memory>
import mna.io.cli_parser;
import mna.io.directory_scanner;
import mna.io.parquet_reader;
import mna.model.iot_network;
import mna.model.job;
import dimna.allocator.di_allocator;
import dimna.model.solution;
import dimna.io.output_writer;

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
    jobsV.push_back(mna::Job{jr, jb, jl, jo});
  };

  pr.read_vertexes(lastInstance.nodes, add_vertex);
  pr.read_edges(lastInstance.edges, add_edge);
  pr.read_jobs(lastInstance.jobs[0], add_job);

  int cut_comb_nodes = 8;
  int cut_sol = 2;

  mna::di::DiAllocator allocator(network, cut_sol, cut_comb_nodes);

  std::vector<mna::di::Solution> solutions = allocator.run_mna_jobs(jobsV);

  std::cout << "T:" << solutions.size() << "\n";

  mna::di::OutputWriter ow(dir_config.output_folder);

  int curr_runnings = 0;

  ow.write_output(solutions, jobsV, 0, {cut_sol, cut_comb_nodes, curr_runnings, network->vertex_count()});

  // for (auto& s : solutions){
  //   std::cout << s.of << " [";
  //   for (auto n : s.nodes){
  //     std::cout << n << ", ";
  //   }
  //   std::cout << "]\n";
  //}
  return 0;
}
