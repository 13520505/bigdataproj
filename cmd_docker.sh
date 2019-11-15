#!/usr/bin/env bash
# 1. create Dockerfile
# 2. build image from Dockerfile
# 3. build container from image
sudo docker build -t news_session:latest .
sudo docker run -p 8082:8082 news_session