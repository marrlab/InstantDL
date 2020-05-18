#!/bin/bash

DIR=$(pwd)

rm -rf ${DIR}/build

mkdir -p ${DIR}/build
mkdir -p ${DIR}/build/classification
mkdir -p ${DIR}/build/data_generator
mkdir -p ${DIR}/build/segmentation
mkdir -p ${DIR}/build/evaluation

cp ${DIR}/docker/Dockerfile ${DIR}/build
cp ${DIR}/instantdl/*.py ${DIR}/build
cp ${DIR}/instantdl/config.json ${DIR}/build # you need to pass the right config file

cp ${DIR}/instantdl/classification/*.py ${DIR}/build/classification
cp ${DIR}/instantdl/data_generator/*.py ${DIR}/build/data_generator
cp ${DIR}/instantdl/segmentation/*.py ${DIR}/build/segmentation
cp ${DIR}/instantdl/evaluation/*.py ${DIR}/build/evaluation