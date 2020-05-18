# InstantDL with Docker

Here you can find the information of using InstantDL with docker.


## Build a Docker container

first you need to download the repository. After that, being in the main folder, simply run 

```bash
sudo docker/bin/docker_build_context.sh
sudo docker build --tag=instantdl:0.0.1 build
```

## Prepration the data folder

For runnig the docker, first you need to create the data folder as exaplined in the [examples](../docs/examples). In case, you would like to pretrained weights as well, first create a folder called `logs` under the data folder and simply put the `.hdf5` file in the folders. After that, set all the parameters according to the documentation EXCEPT `path` and `pretrained_weights_path`, and save the parameters as  `config.json` in the data folder.

__IMPORTANT NOTE__: For the config file, in the json file, these parameters `data` and `pretrained_weights_path` should BE ALWAYS: 

```json
{
    ...
	"path": "/data/",
	"pretrained_weights_path": "/data/logs/pretrained_weights_Lung_SemanticSegmentation.hdf5",
    ...
}
```

The reason for this is that the folder will be mounted to the docker with the address `/data/`

## Data Folder Structure

This is a folder structure of the data folder:

```
PATH_TO_DATA
    ├── train                    
    │   ├── image
    │   │    ├── 000003-num1.png
    │   │    ├── .
    │   │    └── 059994-num1.png     
    │   └── groundtruth  
    │        └── groundtruth.csv
    │
    ├── test                    
    │  ├── image
    │  │    ├── 000002-num1.png
    │  │    ├── .
    │  │    └── 009994-num1.png     
    │  └── groundtruth  
    │       └── groundtruth.csv
    ├── logs                    
    │  └── weights.hdf5
    |
    |
    └── config.json
```

## Run Docker container

```bash
docker run -v PATH_TO_DATA/:/data -it instantdl:0.0.1 /bin/bash
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
- [x] add how to create data