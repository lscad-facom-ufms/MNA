set(CMAKE_CXX_STANDARD 23)

add_library(libdimna_sycl OBJECT)

find_package(AdaptiveCpp CONFIG REQUIRED)

find_package(tabulate CONFIG REQUIRED)

# list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/opt/acpp/include")

target_sources(libdimna_sycl
  PUBLIC
  FILE_SET cxx_modules TYPE CXX_MODULES FILES
  allocator/SyclAllocator.mpp
  allocator/SyclAllocator3.mpp
  policy/filter/SoAFilter.mpp
  PRIVATE
  allocator/SyclAllocator.cpp
  allocator/SyclAllocator3.cpp
)
target_compile_options(libdimna_sycl PUBLIC 
  "-I/opt/acpp/include/AdaptiveCpp"
  "--acpp-targets=generic"
)
target_link_libraries(libdimna_sycl
  PRIVATE
  libMNA
  libdimna
)

target_link_libraries(libdimna_sycl
  PRIVATE
  tabulate::tabulate
)
add_sycl_to_target(TARGET libdimna_sycl)

set_target_properties(libdimna_sycl PROPERTIES PREFIX "")
