set(CMAKE_CXX_STANDARD 23)

add_library(libdimna)

find_package(tabulate CONFIG REQUIRED)
# find_package(OpenMP REQUIRED)

target_sources(libdimna
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES FILES
  allocator/constants.mpp
  allocator/DiAllocator.mpp
  allocator/DiRunner.mpp
  io/OutputWriter.mpp
  model/Solution.mpp
  model/Options.mpp
  model/Times.mpp

  policy/latency/DefaultLatency.mpp
  policy/filter/FixedFilter.mpp
  policy/filter/CanonFixedFilter.mpp
  policy/searcher/FixedSearcher.mpp
  # policy/searcher/OpenMPFixedSearcher.mpp
)

target_link_libraries(libdimna
  PRIVATE
  libMNA
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(libdimna PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(libdimna PRIVATE 
 tabulate::tabulate 
)
# target_compile_options(libgraphs PRIVATE -Werror)

set_target_properties(libdimna PROPERTIES PREFIX "")
