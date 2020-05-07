
# Dockerfile for the Face Detection Service

# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

# Update Linux package lists
RUN apt-get update

# Install build tools (gcc etc.)
RUN apt-get install -y build-essential

# Install ops tools
RUN apt-get install -y procps vim

# Install any needed packages specified in requirements.txt
RUN conda install -c anaconda h5py=2.9.0
RUN conda install -c anaconda hdf5=1.10.4
RUN conda install -c anaconda imageio=2.6.1
RUN conda install -c anaconda tensorflow=1.14.0
RUN conda install -c anaconda keras=2.2.4
RUN conda install -c anaconda matplotlib=3.1.1
RUN conda install -c anaconda numpy=1.16.4
RUN conda install -c anaconda python=3.6.7
RUN conda install -c anaconda scikit-image=0.15.0
RUN conda install -c anaconda scikit-learn=0.21.3
RUN conda install -c anaconda scipy=1.3.00

# Copy the current directory contents into the container at /app
COPY . /app
RUN pwd

# compile in a version label so we always can find the source in git
ARG VERSION=unspecified
LABEL InstantDL.version=$VERSION

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
# @TODO: add GPU and cudatoolkit
ENV PYTHONUNBUFFERED TRUE
ENV INSTANT_DL_CONFIG /app/config.json
ENV NUM_WORKER 4

# Run InstantDL code
CMD ["python main.py"]