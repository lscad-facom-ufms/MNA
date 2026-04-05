
set(CMAKE_CXX_STANDARD 23)

set(NLOHMANN_JSON_BUILD_MODULES ON CACHE BOOL "Enable C++20 modules for nlohmann_json" FORCE)

add_library(libMNA)

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

target_sources(libMNA
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES FILES
  model/iot_network.mpp
  model/job.mpp
  io/cli_parser.mpp
  io/ParquetReader.mpp
  io/JsonReader.mpp
  io/directory_scanner.mpp

  PRIVATE
  io/ParquetReader.cpp
  io/JsonReader.cpp
)
target_link_libraries(libMNA PRIVATE 
  Arrow::arrow_static
  Parquet::parquet_static
  nlohmann_json::nlohmann_json)

target_link_libraries(libMNA
  PRIVATE
  libgraphs
)

set_target_properties(libMNA PROPERTIES PREFIX "")
