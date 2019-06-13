FROM raspbian/stretch
WORKDIR "/app"
RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get install -y git autoconf automake libtool g++ build-essential gcc lsb-release sed sudo tar udev wget python3-setuptools python-setuptools python-pip python3-pip libusb-1.0-0-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev git automake byacc lsb-release cmake libgflags-dev libgoogle-glog-dev liblmdb-dev swig3.0 graphviz libxslt-dev libxml2-dev gfortran python3-dev python-pip python3-pip python3-setuptools python3-markdown python3-pillow python3-yaml python3-pygraphviz python3-h5py python3-nose python3-lxml python3-matplotlib python3-numpy python3-protobuf python3-dateutil python3-skimage python3-scipy python3-six python3-networkx python3-tk mosquitto && apt-get clean 
RUN git clone -b ncsdk2 http://github.com/Movidius/ncsdk 
RUN pip install --upgrade pip 
RUN pip install setuptools 
RUN cd ncsdk && make install && echo 'y' | ./install-opencv.sh
RUN rm -r ncsdk
RUN rm -r ~/opencv* 
COPY ./src /app
COPY ./requirements.txt /app
RUN pip3 install -r requirements.txt
RUN git clone -b ncsdk2 http://github.com/Movidius/ncappzoo
RUN cd ncappzoo/apps/video_objects_scalable && make all
#RUN cd ncappzoo/apps/live-image-classifier && make all
#RUN cd ncappzoo/caffe && make all 
#RUN cd ncappzoo/caffe/GenderNet && make all 
#RUN cd ncappzoo/caffe/AgeNet && make all 
RUN rm -r ncappzoo
#RUN apt-get install -y libgst*
#Run apt-get install -y build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran python2.7-dev python3-dev
CMD ["python3","/app/main.py"]
