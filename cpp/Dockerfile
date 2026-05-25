# Use the official CUDA image as the base image

FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS dev

WORKDIR /app

RUN apt-get update && apt-get install -y \
  software-properties-common \
  git \
  build-essential \
  ninja-build \
  curl \
  zip \
  unzip \
  tar \
  flex \
  bison \
  pkg-config \
  wget \
  inotify-tools \
  clang \
  libclang-dev \
  lld \
  llvm-dev \
  clang-tools

RUN ln -s /usr/bin/clang-scan-deps-18 /usr/bin/clang-scan-deps

RUN wget https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
  && apt-get install -y \ 
  ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
  && apt-get update \
  && apt-get install -y libarrow-dev libparquet-dev libarrow-dataset-dev \
  && rm apache-arrow-apt-source-latest-*.deb

ARG CMAKE_VERSION=4.3.1

RUN curl -L "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -o cmake.tar.gz \
    && tar -xzf cmake.tar.gz -C /usr/local --strip-components=1 \
    && rm cmake.tar.gz

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git && \
    cd AdaptiveCpp && \
    mkdir build && cd build && \
    cmake .. -DWITH_CUDA_BACKEND=ON -DCMAKE_INSTALL_PREFIX=/opt/acpp \
    -DCUDAToolkit_LIBRARY_ROOT=/usr/lib/nvidia-cuda-toolkit \
    -DCUDA_DEVICE_LIBS_PATH=/usr/lib/nvidia-cuda-toolkit/libdevice && \
    make -j && \
    make install
# Setting up vcpkg
WORKDIR /opt/

RUN git clone https://github.com/microsoft/vcpkg.git && ./vcpkg/bootstrap-vcpkg.sh

WORKDIR /app

RUN apt install -y libomp-dev

RUN mkdir -p /usr/lib/nvidia-cuda-toolkit/libdevice/

RUN ln -s /usr/local/cuda/nvvm/libdevice/libdevice.10.bc /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc

RUN apt install -y python3
