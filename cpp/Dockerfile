# Use the official Ubuntu image as the base image
FROM ubuntu:latest

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
  inotify-tools

RUN wget https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
  && apt-get install -y \ 
  ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
  && apt-get update \
  && apt-get install -y libarrow-dev libparquet-dev libarrow-dataset-dev \
  && rm apache-arrow-apt-source-latest-*.deb

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y \
  && apt-get update \
  && apt-get install -y gcc-15 g++-15 \
  && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 100 \
 && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-15 100 \
 && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-15 100

ARG CMAKE_VERSION=4.3.1

RUN curl -L "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -o cmake.tar.gz \
    && tar -xzf cmake.tar.gz -C /usr/local --strip-components=1 \
    && rm cmake.tar.gz

# Setting up vcpkg
WORKDIR /opt/

RUN git clone https://github.com/microsoft/vcpkg.git && ./vcpkg/bootstrap-vcpkg.sh

WORKDIR /app
