add_executable(canon_di_mna)

target_sources(canon_di_mna
  PRIVATE
  main.cpp
  PRIVATE
  # FILE_SET cxx_modules TYPE CXX_MODULES FILES
)
target_link_libraries(canon_di_mna
  PRIVATE
  libMNA
  libdimna
)
