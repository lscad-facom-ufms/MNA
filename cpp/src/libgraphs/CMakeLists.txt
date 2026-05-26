
set(CMAKE_CXX_STANDARD 23)

add_library(libgraphs)

target_sources(libgraphs
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES FILES
  model/Graph.mpp
)

# target_compile_options(libgraphs PRIVATE -Werror)

set_target_properties(libgraphs PROPERTIES PREFIX "")
