#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include <string>

import mna.io.cli_parser;
import mna.io.directory_scanner;
import mna.io.parquet_reader;
import mna.io.json_reader;
import mna.model.iot_network;
import mna.model.job;

import mna.model.soa;

import dimna.allocator.sycl_allocator;
import dimna.allocator.sycl_allocator3;

import dimna.model.solution;
import dimna.model.options;
import dimna.io.output_writer;

import dimna.allocator.di_runner;
import dimna.allocator.di_allocator;

mna::di::Options setup_options(int argc, char* argv[]);
std::shared_ptr<mna::IoTNetworkSoA> setup_network(mna::InstanceStructure& instance, mna::ParquetReader& pr);
mna::JobVector setup_jobs(std::filesystem::path& jobs_file, mna::ParquetReader& pr);

void run(std::shared_ptr<mna::IoTNetworkSoA> network, mna::JobVector& jobs, mna::di::Options di_options);
 
struct JsonNetwork{
  std::shared_ptr<mna::IoTNetworkSoA> network;
  mna::JobVector jobs;
};

JsonNetwork
json_network(std::filesystem::path instance_file, mna::JsonReader& jsonReader);


int
main (int argc, char *argv[]) {

  std::cout << "É o print pô\n";

  mna::di::Options di_options = setup_options(argc, argv);

  mna::InstancesMap instances_map = mna::scan_input_dir(di_options.input_dir);
  
  mna::ParquetReader pr;

  // Looping on different networks present at the input folder
  for (auto& [key, instance] : instances_map){

    auto network = setup_network(instance, pr);

    // Looping through jobs configurantions (light and heavy)
    for (auto& jobs_file : instance.jobs){
      mna::JobVector jobs = setup_jobs(jobs_file, pr);
      
      run(network, jobs, di_options);
    }
  }
  mna::JsonReader jsonReader;

  auto json_instances = mna::scan_for_json(di_options.input_dir);

  for (auto& instance : json_instances){

    auto [network, jobs] = json_network(instance, jsonReader);

    run(network, jobs, di_options);
  }

  return 0;
}

void 
run(std::shared_ptr<mna::IoTNetworkSoA> network, mna::JobVector& jobs, mna::di::Options di_options){

  switch (di_options.cut_sol) {
    case 2:{
      
      mna::di::DiRunner<mna::di::par::SyclAllocator> runner(network, di_options);
      runner.run_mna_jobs_batch(jobs);
    }
    break;

    case 3:{

      mna::di::DiRunner<mna::di::par::SyclAllocator3> runner(network, di_options);

      runner.run_mna_jobs_batch(jobs);
    }
    break;
  }
  
}

mna::di::Options
setup_options(int argc, char* argv[]){
  
  std::string str_cs;
  std::string str_ccn;
  std::string str_nR;
  std::string str_jobs;

  mna::OptionsMap map = {{"--cs", &str_cs},
                        {"--ccn", &str_ccn},
                        {"--runs", &str_nR},
                        {"--jobs", &str_jobs}};

  mna::Config config = mna::read_CLI(argc, argv, map);

  int cut_sol = std::stoi(str_cs);
  int cut_comb_nodes = std::stoi(str_ccn);
  int numRunnings = std::stoi(str_nR);
  
  int sample_jobs;
  try
  {
    sample_jobs = std::stoi::(str_jobs);
  }
  catch(const std::exception& e)
  {
    sample_jobs = 0;
  }
  

  return {config.input_folder, config.output_folder, cut_sol, cut_comb_nodes, numRunnings, sample_j};
}

JsonNetwork
json_network(std::filesystem::path instance_file, mna::JsonReader& jsonReader){

  int total_vertexes = jsonReader.get_num_vertexes(instance_file);

  auto network = std::make_shared<mna::IoTNetworkSoA>(total_vertexes);
  
  using vertex_type = decltype(network)::element_type::vertex_type;
  using edge_type = decltype(network)::element_type::edge_type;

  auto add_edge = [&](int32_t source, int32_t target, int32_t latency){
    network->add_edge(edge_type{target, latency}, source); 
  };
  auto add_vertex = [&](int64_t R, int64_t B, int64_t busy, int64_t inactive){
    network->add_vertex(vertex_type{(int)R, (int)B, (int)busy, (int)inactive});
  };
  
  mna::JobVector jobsV;

  auto add_job = [&](int64_t jr, int64_t jb, int64_t jl, int64_t jo){
    jobsV.push_back(mna::Job{(int)jr, (int)jb, (int)jl, (int)jo});
  };

  jsonReader.read_instance(instance_file, add_vertex, add_edge, add_job);

  return {network, jobsV};
}

std::shared_ptr<mna::IoTNetworkSoA>
setup_network(mna::InstanceStructure& instance, mna::ParquetReader& pr){

  int total_vertexes = *(pr.get_num_rows(instance.nodes));

  auto network = std::make_shared<mna::IoTNetworkSoA>(total_vertexes);

  using vertex_type = decltype(network)::element_type::vertex_type;
  using edge_type = decltype(network)::element_type::edge_type;

  auto add_edge = [&](int32_t source, int32_t target, int32_t latency){
    network->add_edge(edge_type{target, latency}, source); 
  };
  auto add_vertex = [&](int64_t R, int64_t B, int64_t busy, int64_t inactive){
    network->add_vertex(vertex_type{(int)R, (int)B, (int)busy, (int)inactive});
  };
  
  pr.read_vertexes(instance.nodes, add_vertex);
  pr.read_edges(instance.edges, add_edge);

  return network;
}

mna::JobVector
setup_jobs(std::filesystem::path& jobs_file, mna::ParquetReader& pr){

  mna::JobVector jobsV;

  auto add_job = [&](int64_t jr, int64_t jb, int64_t jl, int64_t jo){
    jobsV.push_back(mna::Job{(int)jr, (int)jb, (int)jl, (int)jo});
  };

  pr.read_jobs(jobs_file, add_job);

  return jobsV;
}
