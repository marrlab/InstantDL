#!/usr/bin/env bash
cd malis
python setup.py build_ext --inplace
printf "BUILD COMPLETE\n"
cd ..