set(CMAKE_CXX_STANDARD 23)

add_library(libdimna)

find_package(tabulate CONFIG REQUIRED)

target_sources(libdimna
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES FILES
  allocator/DiAllocator.mpp
  allocator/DiRunner.mpp
  io/OutputWriter.mpp
  model/Solution.mpp
  model/Options.mpp
  policy/searcher/FixedSearcher.mpp
)

target_link_libraries(libdimna
  PRIVATE
  libMNA
)

target_link_libraries(libdimna PRIVATE 
 tabulate::tabulate 
)
# target_compile_options(libgraphs PRIVATE -Werror)

set_target_properties(libdimna PROPERTIES PREFIX "")
