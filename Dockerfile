#Download base image ubuntu 16.04
FROM ubuntu:16.04
FROM python:3.6
# Update Software repository
RUN apt-get update

# Install pip3
RUN apt-get -y install python3-pip
RUN apt-get -y install vim

# Update Software repository
RUN apt-get update

RUN mkdir /news_sessioned

WORKDIR  /news_sessioned

COPY . /news_sessioned

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8082

ENTRYPOINT ["python3","run.py"]
