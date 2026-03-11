#include <cstdint>
#include <iostream>
#include <filesystem>
#include <vector>
import mna.io.parquet_reader;

using namespace std;
using namespace std::filesystem;
int 
main (int argc, char *argv[]) {
  path nodes_path("/home/luige/Projects/graphs/test/MNAexample/input_10nds_nodes.parquet");
  path jobs_path("/home/luige/Projects/graphs/test/MNAexample/input_10nds_2jobs_light.parquet");
  vector<path> edges_path;
  edges_path.push_back(path("/home/luige/Projects/graphs/test/MNAexample/input_10nds_edges_part0.parquet"));

  mna::ParquetReader pr;

  auto print_nodes = [](int64_t r, int64_t b, int64_t busy, int64_t inactive){
    cout << r << ", " << b << ", " << busy << ", " << inactive << "\n";
  };

  auto print_jobs = [](int64_t r, int64_t b, int64_t l, int64_t o){
    cout << r << ", " << b << ", " << l << ", " << o << "\n";
  };
  
  auto print_edges = [](int64_t s, int64_t t, int64_t l){
    cout << s << ", " << t << ", " << l << "\n";
  };

  cout << "Nodes:\n";
  pr.read_vertexes(nodes_path, print_nodes);
  cout << "\nEdges:\n";
  pr.read_edges(edges_path, print_edges);
  cout << "\nJobs:\n";
  pr.read_jobs(jobs_path, print_jobs);

  auto rows = pr.get_num_rows(nodes_path);

  cout << "\nNumber of nodes: " << *(rows) << "\n";
  return 0;
}
