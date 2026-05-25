add_executable(di_mna_sycl)

find_package(AdaptiveCpp CONFIG REQUIRED)

target_sources(di_mna_sycl
  PRIVATE
  main.cpp

  PRIVATE
  # FILE_SET cxx_modules TYPE CXX_MODULES FILES
)
target_compile_options(di_mna_sycl PUBLIC 
  "-I/opt/acpp/include/AdaptiveCpp"
  "--acpp-targets=generic"
)
target_link_libraries(di_mna_sycl
  PRIVATE
  # "-Wl,--whole-archive"
  libdimna_sycl
  # "-Wl,--no-whole-archive"
  libMNA
)
add_sycl_to_target(TARGET di_mna_sycl)
