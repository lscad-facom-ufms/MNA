cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
  "451f2fe2-a8a2-47c3-bc32-94786d8fc91b"
)

project(graphs VERSION 0.0.1 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_MODULE_STD 1)

set(CMAKE_CXX_STANDARD 23)

option(ENABLE_COVERAGE "Activate flags for gcov" OFF)
option(ENABLE_GPROG "Activate gprof" OFF)
if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

if(ENABLE_GPROF)
    add_compile_options(-pg -O3 -g) 
    add_link_options(-pg)
else()
  add_compile_options(-O3)
endif()

add_subdirectory(libgraphs)
add_subdirectory(libMNA)
add_subdirectory(libdimna)
add_subdirectory(dimna)
add_subdirectory(canon_dimna)
# add_subdirectory(openmp_fixed_dimna)

# if (BUILD_TESTING)
  # add_subdirectory(test)
# endif()
