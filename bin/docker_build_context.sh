#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..

rm -rf ${DIR}/build/docker

mkdir -p ${DIR}/build/docker
mkdir -p ${DIR}/build/docker/classification
mkdir -p ${DIR}/build/docker/data_generator
mkdir -p ${DIR}/build/docker/segmentation
mkdir -p ${DIR}/build/docker/evaluation


cp ${DIR}/Dockerfile ${DIR}/build/docker
cp ${DIR}/*.py ${DIR}/build/docker
cp ${DIR}/config.json ${DIR}/build/docker

cp ${DIR}/classification/*.py ${DIR}/build/docker/classification
cp ${DIR}/data_generator/*.py ${DIR}/build/docker/data_generator
cp ${DIR}/data_generator/*.py ${DIR}/build/docker/segmentation
cp ${DIR}/data_generator/*.py ${DIR}/build/docker/evaluation