#include <filesystem>
#include <memory>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
import mna.io.json_reader;

namespace mna {

namespace fs = std::filesystem;
using json = nlohmann::json;

int
JsonReader::get_num_vertexes(fs::path instance_file){

  std::ifstream f(instance_file);
  json data = json::parse(f);

  std::vector<int64_t> V_R = data["V_R"].get<std::vector<int64_t>>();

  return V_R.size();
  
}

void
JsonReader::read_instance(fs::path instance_file, VertexCallback consume_vertex, EdgeCallback consume_edge, JobCallback consume_job){
  std::ifstream f(instance_file);
  json data = json::parse(f);

  if (!data.contains("jr") || !data["jr"].is_array())
    throw std::runtime_error("Missing 'jr' field on JSON file: " + instance_file.string()); 

  if (!data.contains("jb") || !data["jb"].is_array())
    throw std::runtime_error("Missing 'jb' field on JSON file: " + instance_file.string()); 
  
  if (!data.contains("jl") || !data["jl"].is_array())
    throw std::runtime_error("Missing 'jl' field on JSON file: " + instance_file.string()); 
  
  if (!data.contains("jo") || !data["jo"].is_array())
    throw std::runtime_error("Missing 'jo' field on JSON file: " + instance_file.string());

  if (!data.contains("V_R") || !data["V_R"].is_array())
    throw std::runtime_error("Missing 'V_R' field on JSON file: " + instance_file.string()); 

  if (!data.contains("V_B") || !data["V_B"].is_array())
    throw std::runtime_error("Missing 'V_B' field on JSON file: " + instance_file.string()); 
  
  if (!data.contains("adjList") || !data["adjList"].is_array())
    throw std::runtime_error("Missing 'adjList' field on JSON file: " + instance_file.string()); 
  
  {
    std::vector<int64_t> V_R = data["V_R"].get<std::vector<int64_t>>();
    std::vector<int64_t> V_B = data["V_B"].get<std::vector<int64_t>>();

    for (int i = 0; i < V_R.size(); ++i){
      consume_vertex(V_R[i], V_B[i], 0, 0);
    }
  }
  
  {
    std::vector<std::vector<std::vector<int>>> adjList = data["adjList"];

    for (int i = 0; i < adjList.size(); ++i){
      for (int j = 0; j < adjList[i].size(); ++j){
        consume_edge(i, adjList[i][j][0], adjList[i][j][1]);
      }
    }
  }

  {
    std::vector<int64_t> jr = data["jr"].get<std::vector<int64_t>>();
    std::vector<int64_t> jb = data["jb"].get<std::vector<int64_t>>();
    std::vector<int64_t> jl = data["jl"].get<std::vector<int64_t>>();
    std::vector<int64_t> jo = data["jo"].get<std::vector<int64_t>>();

    for (int i = 0; i < jr.size(); ++i){
      consume_job(jr[i], jb[i], jl[i], jo[i]);
    }
  }
}
}
