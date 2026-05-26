#include <string>
#include <fstream>
#include <filesystem>
#include <exception>
import mna.io.directory_scanner;

namespace fs = std::filesystem;

bool
success_test(fs::path& temp_dir){
  
  fs::path success = temp_dir / "success";
  fs::create_directory(success);

  std::ofstream(success/"input_99nds_5jobs_light.parquet").put('\0');
  std::ofstream(success/"input_99nds_5jobs_light.info.parquet").put('\0');
  std::ofstream(success/"input_99nds_edges_part0.parquet").put('\0');
  std::ofstream(success/"input_99nds_edges_manifest.parquet").put('\0');
  std::ofstream(success/"input_99nds_nodes.parquet").put('\0');
  
  try {
    auto instances = mna::scan_input_dir(success);   
  }
  catch (const std::exception& e) {
    fs::remove_all(success);
    throw e;
  }
  
  fs::remove_all(success);

  return true;
}

bool
manifest_test(fs::path& temp_dir){
  
  fs::path missing_manifest = temp_dir / "missing_manifest";
  fs::create_directory(missing_manifest);

  std::ofstream(missing_manifest/"input_99nds_5jobs_light.parquet").put('\0');
  std::ofstream(missing_manifest/"input_99nds_5jobs_light.info.parquet").put('\0');
  std::ofstream(missing_manifest/"input_99nds_edges_part0.parquet").put('\0');
  std::ofstream(missing_manifest/"input_99nds_nodes.parquet").put('\0');
  
  try {
    auto instances = mna::scan_input_dir(missing_manifest);   
    fs::remove_all(missing_manifest);
    return false;
  }
  catch (const std::exception& e) {
    fs::remove_all(missing_manifest);
    return true;
  }

}

bool
nodes_test(fs::path& temp_dir){
  
  fs::path success = temp_dir / "missing_nodes";
  fs::create_directory(success);

  std::ofstream(success/"input_99nds_5jobs_light.parquet").put('\0');
  std::ofstream(success/"input_99nds_5jobs_light.info.parquet").put('\0');
  std::ofstream(success/"input_99nds_edges_part0.parquet").put('\0');
  std::ofstream(success/"input_99nds_edges_manifest.parquet").put('\0');
  
  try {
    auto instances = mna::scan_input_dir(success);   
    fs::remove_all(success);
    return false;
  }
  catch (const std::exception& e) {
    fs::remove_all(success);
    return true;
  }
}
bool
edges_test(fs::path& temp_dir){
  
  fs::path success = temp_dir / "edges";
  fs::create_directory(success);

  std::ofstream(success/"input_99nds_5jobs_light.parquet").put('\0');
  std::ofstream(success/"input_99nds_5jobs_light.info.parquet").put('\0');
  std::ofstream(success/"input_99nds_edges_manifest.parquet").put('\0');
  std::ofstream(success/"input_99nds_nodes.parquet").put('\0');
  
  try {
    auto instances = mna::scan_input_dir(success);   
    fs::remove_all(success);
    return false;
  }
  catch (const std::exception& e) {
    fs::remove_all(success);
    return true;
  }
}
bool
jobs_test(fs::path& temp_dir){
  
  fs::path success = temp_dir / "success";
  fs::create_directory(success);

  std::ofstream(success/"input_99nds_5jobs_light.info.parquet").put('\0');
  std::ofstream(success/"input_99nds_edges_part0.parquet").put('\0');
  std::ofstream(success/"input_99nds_edges_manifest.parquet").put('\0');
  std::ofstream(success/"input_99nds_nodes.parquet").put('\0');
  
  try {
    auto instances = mna::scan_input_dir(success);   
    fs::remove_all(success);
    return false;
  }
  catch (const std::exception& e) {
    fs::remove_all(success);
    return true;
  }
}
int
main(){
  
  fs::path temp_dir = fs::temp_directory_path() / "scanner_test";
  fs::remove_all(temp_dir);
  fs::create_directory(temp_dir);

  if(!success_test(temp_dir)){
    throw std::runtime_error("Error: Unsuccessful try!");
  }

  if(!manifest_test(temp_dir)){
    throw std::runtime_error("Error: Missing Manifest file not detected!");
  }

  if(!nodes_test(temp_dir)){
    throw std::runtime_error("Error: Missing Nodes file not detected!");
  }

  if(!edges_test(temp_dir)){
    throw std::runtime_error("Error: Missing Edges file(s) not detected!");
  }

  if(!jobs_test(temp_dir)){
    throw std::runtime_error("Error: Missing Jobs file(s) not detected!");
  }
  fs::remove_all(temp_dir);
  return 0;
}
