# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
#FROM tensorflow/tensorflow:1.4.1-gpu-py3 AS NECESSARY_UP
#
#RUN apt-get update \
#    && apt-get -y upgrade \
#    && apt-get install -y build-essential cmake pkg-config \
#    && apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
#    && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
#    && apt-get install -y libgtk-3-dev \
#    && apt-get install -y libatlas-base-dev gfortran \
#    && apt install -y git wget
#
#FROM NECESSARY_UP AS TF_WITH_OPENCV
#
#RUN cd \
#    && wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip \
#    && unzip opencv_contrib.zip
#
#RUN pwd && ls
#
#COPY resources/opencv-3.1.0 /root/opencv-3.1.0
#
#RUN cd /root/opencv-3.1.0 \
#    && mkdir build \
#    && cd build \
#    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
#             -D CMAKE_INSTALL_PREFIX=/usr/local \
#             -D INSTALL_PYTHON_EXAMPLES=OFF \
#             -D INSTALL_C_EXAMPLES=OFF \
#             -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib-3.1.0/modules \
#             -D PYTHON_EXECUTABLE=/usr/bin/python \
#             -D BUILD_EXAMPLES=OFF ..
#
#FROM TF_WITH_OPENCV AS TF_WITH_OPENCV_INSTALLED
#
#RUN cd /root/opencv-3.1.0/build \
#    && make -j8 \
#    && make install \
#    && ldconfig
#
#RUN cp cv2.cpython-35m-x86_64-linux-gnu.so /usr/local/lib/python3.5/dist-packages/ \
#    && cd /usr/local/lib/python3.5/dist-packages/ \
#    && mv cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
FROM continuumio/miniconda3 AS CONDA_STAGE

WORKDIR /pointnet-autoencoder

RUN /opt/conda/bin/conda create -y -n pointnetae python=3.6

ENV PATH /opt/conda/envs/pointnetae/bin:$PATH


RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y pkg-config \
    && apt-get install -y libgtk2.0-dev \
    && /bin/bash -c "source activate pointnetae \
&& conda install -y opencv"


FROM tensorflow/tensorflow:1.4.1-gpu-py3 AS CONDA_TF_AND_OPENCV_STAGE

COPY --from=CONDA_STAGE /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:/opt/conda/condabin:/opt/conda/envs/pointnetae/bin:$PATH

RUN /bin/bash -c "source activate pointnetae \
&& pip install tensorflow-gpu==1.4"
# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
#FROM tensorflow/tensorflow:1.4.1-gpu-py3 AS CONDA_TF_AND_OPENCV_STAGE
#RUN pip install --upgrade pip \
#    && pip install opencv-python \
#    && pip install opencv-contrib-python
# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#
FROM CONDA_TF_AND_OPENCV_STAGE AS POINTNET_AE_BUILD_STAGE

WORKDIR /pointnet-autoencoder/

CMD /bin/bash -c "source activate pointnetae && python"

COPY tf_ops/ tf_ops
COPY models/ models/
COPY utils/ utils

RUN cd /pointnet-autoencoder/tf_ops/nn_distance \
    && chmod 777 tf_nndistance_compile.sh \
    && sh tf_nndistance_compile.sh

RUN cd /pointnet-autoencoder/utils \
    && chmod 777 compile_render_balls_so.sh \
    && sh compile_render_balls_so.sh
