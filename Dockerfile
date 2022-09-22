FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update -y
RUN apt install python3 -y
RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN pip install keras
RUN pip install tensorflow
RUN pip install Pillow
RUN pip install scipy

RUN mkdir /data
COPY train.py /
COPY data /data

ENTRYPOINT ["python3" ,"train.py"]
CMD ["data/"]
