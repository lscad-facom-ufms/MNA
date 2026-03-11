cmake_minimum_required(VERSION 4.0)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
  "d0edc3af-4c50-42ea-a356-e2862fe7a444"
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
endif()

add_subdirectory(libgraphs)
add_subdirectory(libMNA)
add_subdirectory(libdimna)
add_subdirectory(dimna)

if (BUILD_TESTING)
  add_subdirectory(test)
endif()
