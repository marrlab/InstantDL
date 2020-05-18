#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..

rm -rf ${DIR}/docker/build

mkdir -p ${DIR}/docker/build
mkdir -p ${DIR}/docker/build/classification
mkdir -p ${DIR}/docker/build/data_generator
mkdir -p ${DIR}/docker/build/segmentation
mkdir -p ${DIR}/docker/build/evaluation

cp ${DIR}/docker/Dockerfile ${DIR}/docker/build
cp ${DIR}/instantdl/*.py ${DIR}/docker/build
cp ${DIR}/instantdl/config.json ${DIR}/docker/build # you need to pass the right config file

cp ${DIR}/instantdl/classification/*.py ${DIR}/docker/build/classification
cp ${DIR}/instantdl/data_generator/*.py ${DIR}/docker/build/data_generator
cp ${DIR}/instantdl/segmentation/*.py ${DIR}/docker/build/segmentation
cp ${DIR}/instantdl/evaluation/*.py ${DIR}/docker/build/evaluation