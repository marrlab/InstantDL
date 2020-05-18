# InstantDL with Docker

Here you can find the information of using InstantDL with docker.


## Build a Docker container

first you need to download the repository. After that, being in the main folder, simply run 

```bash
sudo docker/bin/docker_build_context.sh
sudo docker build --tag=instantdl:0.0.1 build
```

## Prepration the data folder

For runnig the docker, you need to create the data folder

## Run Docker container

```bash
docker run -d --name instantdl -p 8000:80 -v <Path_to_Data_Folder>:/app/data
```

For example you would like to run the classifiction example. 

## GPU support

We are using the docker from [tensorflow](https://www.tensorflow.org/install/docker). To Check if a GPU is available:

```bash
lspci | grep -i nvidia
```

Verify your nvidia-docker installation:

```bash
sudo docker run --gpus all --rm nvidia/cuda nvidia-smi
```

### Cleanup

Remove all containers and images:

```bash
docker system prune -a
```

## TODO

- [x] add cuda and gpu