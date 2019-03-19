FROM ubuntu:18.04
ENV SHELL=/bin/bash
RUN apt-get update && apt-get install -y git curl
RUN apt-get install -y python3-setuptools python3-dev build-essential python3-pip && apt-get clean
RUN apt-get install -y libsm6 libxext6 libxrender-dev openslide-tools && apt-get clean 

RUN pip3 install cython numpy pandas jupyter
#RUN pip3 install git+https://github.com/DSLituiev/slideslicer
ARG CACHEBUST=2
RUN git clone https://github.com/dslituiev/slideslicer
RUN cd slideslicer/ && pip3 install .

### DOWNLOAD FILES
RUN mkdir data

COPY 14112.* data/
#RUN fileid="1-s87PeyZY2C6ZejCyn4Cy0iYEkTeEFrP" &&\
#  filename=data/14112.svs &&\
#  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null &&\
#  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename} &&\
#  rm cookie
#
#RUN fileid="1JRBfyzeyMdg0WTc2McC_wOiYrXhi4iRx" &&\
#  filename=data/14112.xml &&\
#  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null && \
#  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename} &&\
#  rm cookie
COPY resnet50_mouse_gloms_scale4.h5 data/

RUN ln -s data slideslicer
RUN pip3 install scikit-learn tensorflow keras

CMD cd slideslicer/notebooks && jupyter-notebook --port 8899 --ip=0.0.0.0 --allow-root

