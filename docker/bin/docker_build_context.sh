#!/bin/bash

DIR=$(pwd)

rm -rf ${DIR}/build

mkdir -p ${DIR}/build/instantdl
#mkdir -p ${DIR}/build/instantdl/classification
#mkdir -p ${DIR}/build/instantdl/data_generator
#mkdir -p ${DIR}/build/instantdl/segmentation
#mkdir -p ${DIR}/build/instantdl/evaluation

cp ${DIR}/docker/Dockerfile ${DIR}/build
cp ${DIR}/setup.py ${DIR}/build/setup.py
cp ${DIR}/README.md ${DIR}/build/README.md

cp -r ${DIR}/instantdl/* ${DIR}/build/instantdl

#cp ${DIR}/instantdl/classification/*.py ${DIR}/build/instantdl/classification
#cp ${DIR}/instantdl/data_generator/*.py ${DIR}/build/instantdl/data_generator
#cp ${DIR}/instantdl/segmentation/*.py ${DIR}/build/instantdl/segmentation
#cp ${DIR}/instantdl/evaluation/*.py ${DIR}/build/instantdl/evaluation