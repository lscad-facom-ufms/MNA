#include <cstdint>
#include <stdexcept>
import mna.io.cli_parser;
import mna.io.directory_scanner;
import mna.io.parquet_reader;
import mna.model.iot_network;

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

  mna::IoTNetwork network(total_vertexes);

  using vertex_type = decltype(network)::vertex_type;
  using edge_type = decltype(network)::edge_type;

  auto add_edge = [&](int32_t source, int32_t target, int32_t latency){
    network.add_edge(edge_type{target, latency}, source); 
  };
  auto add_vertex = [&](int64_t R, int64_t B, int64_t busy, int64_t inactive){
    network.add_vertex(vertex_type{R, B, busy, inactive});
  };

  pr.read_vertexes(lastInstance.nodes, add_vertex);
  pr.read_edges(lastInstance.edges, add_edge);

  if(network.vertex_count() != total_vertexes){
    throw std::runtime_error("Error: Vertexes file miss read.");
  }

  if(network.edge_count() != total_edges){
    throw std::runtime_error("Error: Edges file miss read.");
  }

  return 0;
}
