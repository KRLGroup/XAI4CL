# Docker file for PyTorch
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

USER root

# Set working directory
WORKDIR /workspace

# Update pip
RUN apt-get update
RUN pip install --upgrade pip

# Install Python packages
RUN pip install avalanche-lib==0.5 omegaconf pytorchcv==0.0.67 seaborn==0.13.2