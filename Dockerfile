FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

WORKDIR /pointnet-autoencoder

COPY /resources/Miniconda3-latest-Linux-x86_64.sh /pointnet-autoencoder

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y build-essential cmake pkg-config \
    && apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    && apt-get install -y libxvidcore-dev libx264-dev \
    && apt-get install -y libgtk-3-dev \
    && apt-get install -y libatlas-base-dev gfortran \
    && apt-get install -y pkg-config

RUN chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:${PATH}

RUN conda update -y conda \
    && apt install libgl1-mesa-glx

RUN conda create -n venv python=3.6 \
    && /bin/bash -c "source activate venv \
&& pip install opencv-python \
&& pip install opencv-contrib-python \
&& pip install tensorflow-gpu==1.4"


WORKDIR /pointnet-autoencoder/

COPY tf_ops/ tf_ops
COPY models/ models/
COPY utils/ utils

RUN cd /pointnet-autoencoder/tf_ops/nn_distance \
    && chmod 777 tf_nndistance_compile.sh \
    && sh tf_nndistance_compile.sh

RUN cd /pointnet-autoencoder/tf_ops/approxmatch \
    && chmod 777 tf_approxmatch_compile.sh \
    && sh tf_approxmatch_compile.sh

RUN cd /pointnet-autoencoder/utils \
    && chmod 777 compile_render_balls_so.sh \
    && sh compile_render_balls_so.sh
