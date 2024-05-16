FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.10 \
    python3-pip \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3.10 -m pip install --no-cache-dir --upgrade pip uv==0.1.11 && \
    python3.10 -m uv pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    invisible_watermark && \
    python3.10 -m pip install --no-cache-dir \
    accelerate \
    datasets \
    hf-doc-builder \
    huggingface-hub \
    Jinja2 \
    librosa \
    numpy \
    scipy \
    tensorboard \
    transformers \
    pytorch-lightning \
    matplotlib

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# Set environment variable
ENV CUBLAS_WORKSPACE_CONFIG=:16:8

RUN git clone https://github.com/yousirong/diffusers.git /app
# RUN git config --gloabal credential.helper store



WORKDIR /app
RUN mkdir -p /app/ddpm-butterflies-128 
COPY ./ddpm-butterflies-128 /app/ddpm-butterflies-128 

CMD ["bin/bash"]

# sudo docker build -t diff_tuto docker/diffusers-pytorch-cuda

# new build images
# sudo docker build --no-cache -t diff_tuto docker/diffusers-pytorch-cuda

# sudo docker run --gpus all -it --rm diff_tuto bash

# pip install -e .

# huggingface-cli login

# python examples/unconditional_image_generation/train_unconditional.py --dataset_name huggan/smithsonian_butterflies_subset --train_batch_size 8 --num_epochs 50 --output_dir ddpm-butterflies-128 --push_to_hub --logger tensorboard --logging_dir /app/workdir/logs
# sudo docker cp d037b872f18e:/app/ddpm-butterflies-128 /home/juneyonglee/Desktop

