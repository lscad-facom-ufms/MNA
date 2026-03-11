#include <vector>
#include <string>
#include <stdexcept>

import mna.io.cli_parser;

using namespace std;

bool simple_success();
bool success_extra_option();
bool missing_input();
bool missing_output();
bool missing_input_value();
bool missing_output_value();
bool missing_extra_option_value();
bool help_detected();

int main(){
  if(!simple_success()){
    throw runtime_error("Error: Arguments not detected!");
  }
  if(!success_extra_option()){
    throw runtime_error("Error: Extra options not detected!");
  }
  if(!help_detected()){
    throw runtime_error("Error: Help not detected!");
  }
  if(!missing_input()){
    throw runtime_error("Error: Missing input not detected!");
  }
  if(!missing_output()){
    throw runtime_error("Error: Missing output not detected!");
  }
  if(!missing_input_value()){
    throw runtime_error("Error: Invalid input not detected!");
  }
  if(!missing_output_value()){
    throw runtime_error("Error: Invalid output not detected!");
  }

  return 0;
}

bool
simple_success(){
  vector<const char*> argv = {"test", "--input", "in_dir", "--output", "out_dir"};
  
  mna::Config config;

  config = mna::read_CLI(argv.size(), (char**)argv.data());
  
  return config.input_folder == "in_dir" && config.output_folder == "out_dir";
}

bool
success_extra_option(){

  vector<const char*> argv = {"test", "--input", "in_dir", "--out", "out_dir", "--cut_sol", "2"};
  
  string cut_sol_str;
  mna::OptionsMap opt = {{"--cut_sol", &cut_sol_str}};

  mna::Config config = mna::read_CLI(argv.size(), (char**)argv.data(), opt);
  
  return config.input_folder == "in_dir" && config.output_folder == "out_dir" && cut_sol_str == "2";
}

bool
help_detected(){
  vector<const char*> argv = {"test", "--help"};

  mna::Config config = mna::read_CLI(argv.size(), (char**)argv.data());
  
  return config.showHelp;
}
bool
missing_input(){

  vector<const char*> argv = {"test", "--output", "out_dir"};

  try {
    mna::Config config = mna::read_CLI(argv.size(), (char**)argv.data());
  } catch (const std::exception& e) {
    return true;
  }
  return false;
}

bool
missing_output(){

  vector<const char*> argv = {"test", "--input", "in_dir"};
  
  try {
    mna::Config config = mna::read_CLI(argv.size(), (char**)argv.data());
  } catch (const std::exception& e) {
    return true;
  }
  return false;
}

bool
missing_input_value(){

  vector<const char*> argv = {"test", "--in"};
  
  try {
    mna::Config config = mna::read_CLI(argv.size(), (char**)argv.data());
  } catch (const std::exception& e) {
    return true;
  }
  return false;
}
bool
missing_output_value(){

  vector<const char*> argv = {"test", "--out"};
  
  try {
    mna::Config config = mna::read_CLI(argv.size(), (char**)argv.data());
  } catch (const std::exception& e) {
    return true;
  }
  return false;
}
bool
missing_extra_option_value(){

  vector<const char*> argv = {"test", "--input", "in_dir", "--out", "out_dir", "--cut_sol"};
  
  string cut_sol_str;
  mna::OptionsMap opt = {{"--cut_sol", &cut_sol_str}};

  try {
    mna::Config config = mna::read_CLI(argv.size(), (char**)argv.data(), opt);
  } catch (const std::exception& e) {
    return true;
  }
  return false;
}

