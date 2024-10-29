FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Asia/Shanghai
ENV TIME_ZONE=${TZ}

COPY ubuntu22.sources.list /etc/apt/sources.list

# 安装必要工具
RUN apt update && apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        tzdata \
        ca-certificates \
        git \
        curl \
        wget \
        vim \
        gdb \
        iputils-ping \
        net-tools \
        lsb-release \
        libnuma-dev \
        ibverbs-providers \
        librdmacm-dev \
        ibverbs-utils \
        rdmacm-utils \
        libibverbs-dev \
        libtinfo-dev \
        libedit-dev \
        libxml2-dev \
        libssl-dev \
        openssl \
        libffi-dev \
    && ln -snf /usr/share/zoneinfo/${TIME_ZONE} /etc/localtime \
    && echo ${TIME_ZONE} > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt clean && apt autoclean && apt autoremove -y \
    && rm -rf /tmp/* /var/cache/* /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/*

# 编译安装cmake
RUN cd /tmp && wget https://cmake.org/files/v3.25/cmake-3.25.3.tar.gz \
    && tar -xzf cmake-3.25.3.tar.gz && cd cmake-3.25.3 \
    && ./configure && make -j8 && make install \
    && rm -rf /tmp/cmake-3.25.3*

# 安装nsight system cli以便后面进行性能分析
RUN apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2204/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt update \
    && apt install -y nsight-systems-cli-2024.5.1 \
    && apt clean && apt autoclean && apt autoremove -y

# 编译安装python3.11
ARG PY_INSTALL_PREFIX="/usr/local"
RUN apt install -y zlib1g zlib1g-dev libsqlite3-dev \
    && cd /tmp && wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz \
    && tar -xzf Python-3.11.9.tgz && cd Python-3.11.9 \
    && ./configure --enable-shared --enable-optimizations --with-zlib --enable-loadable-sqlite-extensions \
         --prefix=${PY_INSTALL_PREFIX} \
    && make -j8 && make install && ldconfig \
    && ln -s ${PY_INSTALL_PREFIX}/bin/pip3 ${PY_INSTALL_PREFIX}/bin/pip \
    && ln -s ${PY_INSTALL_PREFIX}/bin/python3 ${PY_INSTALL_PREFIX}/bin/python \
    && rm -rf /tmp/Python-3.11.9* \
    && apt clean && apt autoclean && apt autoremove -y
COPY pip.conf /root/.pip/pip.conf

RUN pip install -U pip pygwalker seaborn scikit-learn ipykernel