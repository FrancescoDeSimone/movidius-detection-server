FROM raspbian/stretch
WORKDIR "/app"
RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get install -y git autoconf automake libtool g++ build-essential gcc lsb-release sed sudo tar udev wget python3-setuptools python-setuptools python-pip python3-pip libusb-1.0-0-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev git automake byacc lsb-release cmake libgflags-dev libgoogle-glog-dev liblmdb-dev swig3.0 graphviz libxslt-dev libxml2-dev gfortran python3-dev python-pip python3-pip python3-setuptools python3-markdown python3-pillow python3-yaml python3-pygraphviz python3-h5py python3-nose python3-lxml python3-matplotlib python3-numpy python3-protobuf python3-dateutil python3-skimage python3-scipy python3-six python3-networkx python3-tk mosquitto && apt-get clean 
RUN git clone -b ncsdk2 http://github.com/Movidius/ncsdk 
RUN pip3 install --upgrade pip 
RUN pip3 install setuptools 
RUN pip3 install tensorflow==1.11.0
RUN cd ncsdk && ./install.sh 
RUN cd ncsdk && echo 'y' | ./install-opencv.sh
RUN rm -r ncsdk
RUN rm -r ~/opencv* 
COPY ./src /app
COPY ./requirements.txt /app
RUN pip3 install --no-cache-dir paho-mqtt 
RUN pip3 install --no-cache-dir imutils 
CMD ["python3","/app/main.py"]
