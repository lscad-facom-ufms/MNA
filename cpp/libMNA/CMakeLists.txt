
set(CMAKE_CXX_STANDARD 23)

add_library(libMNA)

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

target_sources(libMNA
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES FILES
  model/iot_network.mpp
  model/job.mpp
  io/cli_parser.mpp
  io/ParquetReader.mpp
  io/directory_scanner.mpp

  PRIVATE
  io/ParquetReader.cpp
)
target_link_libraries(libMNA PRIVATE 
  Arrow::arrow_static
  Parquet::parquet_static)

target_link_libraries(libMNA
  PRIVATE
  libgraphs
)

set_target_properties(libMNA PROPERTIES PREFIX "")
