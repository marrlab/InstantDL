
# Dockerfile for the Face Detection Service

# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Set the working directory to /app
WORKDIR /app

# Update Linux package lists
RUN apt-get update

# Install build tools (gcc etc.)
RUN apt-get install -y build-essential

# Install ops tools
RUN apt-get install -y procps vim

# Install any needed packages specified in requirements.txt
RUN pip install h5py==2.9.0
#RUN pip install hdf5==1.10.4
RUN pip install imageio==2.6.1
RUN pip install keras==2.2.4
RUN pip install matplotlib==3.1.1
RUN pip install numpy==1.16.4
RUN pip install pandas==1.0.3
RUN pip install scikit-image==0.15.0
RUN pip install scikit-learn==0.21.3
RUN pip install scipy==1.3.0

# Copy the current directory contents into the container at /app
COPY . /app
RUN pwd

RUN python setup.py install

# compile in a version label so we always can find the source in git
ARG VERSION=unspecified
LABEL instantdl.version=$VERSION

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
# @TODO: add GPU and cudatoolkit
ENV PYTHONUNBUFFERED TRUE
ENV INSTANT_DL_CONFIG /data/config.json
ENV NUM_WORKER 4

# Run InstantDL code
CMD ["python", "instantdl/main.py", "--config", "/data/config.json"]