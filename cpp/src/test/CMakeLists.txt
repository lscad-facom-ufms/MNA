
add_executable(cliTest)

target_sources(cliTest
  PRIVATE
  unit/cli_parser.test.cpp)

target_link_libraries(cliTest
  PRIVATE
  libMNA
)
add_test(NAME cli COMMAND cliTest)

add_executable(dsTest)

target_sources(dsTest
  PRIVATE
  unit/directory_scanner.test.cpp)

target_link_libraries(dsTest
  PRIVATE
  libMNA
)

add_test(NAME directory_scanner COMMAND dsTest)

add_executable(parquetReaderTest)

target_sources(parquetReaderTest
  PRIVATE
  unit/parquet_reader.test.cpp)

target_link_libraries(parquetReaderTest
  PRIVATE
  libMNA
)

add_test(NAME parquet_reader COMMAND parquetReaderTest)

add_executable(readNetworkTest)

target_sources(readNetworkTest
  PRIVATE
  read_network/main.test.cpp)

target_link_libraries(readNetworkTest
  PRIVATE
  libMNA
  libgraphs
)

add_test(NAME read_network 
  COMMAND readNetworkTest "--input" "${CMAKE_CURRENT_SOURCE_DIR}/MNAexample"
  "--output" "${CMAKE_SOURCE_DIR}/output")

add_executable(latTest)

target_sources(latTest
  PRIVATE
  unit/latencies.test.cpp)

target_link_libraries(latTest
  PRIVATE
  libdimna
)

add_test(NAME latencies 
  COMMAND latTest "--input" "${CMAKE_CURRENT_SOURCE_DIR}/MNAexample"
  "--output" "${CMAKE_SOURCE_DIR}/output")

add_executable(run_mnaTest)

target_sources(run_mnaTest
  PRIVATE
  unit/run_mna.test.cpp)

target_link_libraries(run_mnaTest
  PRIVATE
  libdimna
)

add_test(NAME run_mna_jobs
  COMMAND latTest "--input" "${CMAKE_CURRENT_SOURCE_DIR}/MNAexample"
  "--output" "${CMAKE_SOURCE_DIR}/output")
