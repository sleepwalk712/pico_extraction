FROM langchain/langchain

WORKDIR /app

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  cuda-toolkit-12-4

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

RUN useradd -ms /bin/bash user
USER user

COPY . /app

EXPOSE 8888
