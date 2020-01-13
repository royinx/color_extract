ARG cuda_version=10.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel


# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ \
      git \
      libsm6 \
      libxext6 \
      libxrender1 \
      libglib2.0-0 \
      python3-pip\
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python \
				 numpy \
				 scikit-learn \
				 matplotlib \