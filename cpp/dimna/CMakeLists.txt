add_executable(di_mna)

target_sources(di_mna
  PRIVATE
  main.cpp
  PRIVATE
  # FILE_SET cxx_modules TYPE CXX_MODULES FILES
)
target_link_libraries(di_mna
  PRIVATE
  libMNA
  libdimna
)
