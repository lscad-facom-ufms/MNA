# acpp_toolchain.cmake

# 1. O compilador volta a ser o oficial (para que os testes do CMake passem)
set(CMAKE_CXX_COMPILER "/opt/acpp/bin/acpp" CACHE FILEPATH "AdaptiveCpp C++ compiler")

# 2. O Scanner de dependências agora é o nosso filtro!
set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "/tmp/scan_wrapper.sh" CACHE FILEPATH "Scanner Filtrado")

# 3. O resto continua igual
set(AdaptiveCpp_DIR "/opt/acpp/lib/cmake/AdaptiveCpp" CACHE PATH "Path to AdaptiveCpp Config")
set(AdaptiveCpp_ROOT "/opt/acpp" CACHE PATH "AdaptiveCpp Root")
