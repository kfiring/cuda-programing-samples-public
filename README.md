# cuda-programing-samples-public

本仓库代码是与《CUDA编程基础》系列文章配套的代码，欢迎阅读原文。  
使用方式：  
1. clone本仓库到本地：
```shell
# <local_dir>指存放本仓库代码的本地目录，替换为自己的本地路径
cd <local_dir>
git clone --recursive https://github.com/kfiring/cuda-programing-samples-public.git
```
2. 构建docker镜像（如果没有安装docker，请先安装好docker环境，可以参考[Docker — 从入门到实践](https://yeasy.gitbook.io/docker_practice)进行安装）
```shell
# 注意将<local_dir>替换为自己的本地路径
cd <local_dir>/cuda-programing-samples-public/docker
docker build -t ubt22-cuda118-py311 .
```
3. 启动docker容器
```shell
# 注意将<local_dir>替换为自己的本地路径
docker run --name cuda-prog --privileged --gpus=all  --cap-add=SYS_ADMIN --rm \
    -v "<local_dir>/cuda-programing-samples-public:/code/cuda-programing-samples-public" \
    -itd ubt22-cuda121-py311 /bin/bash -c "sleep 100000000"
```
4. 登录进入docker容器并编译代码
```shell
docker exec -it cuda-prog /bin/bash
# 以下在docker容器中执行
cd /code/cuda-programing-samples-public
mkdir build && cd build
cmake ..
make
```
5. 如果要开发调试代码，则可以使用vscode并安装必要插件后即可连接到docker容器进行开发