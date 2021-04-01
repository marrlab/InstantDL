# Segmentation 

In this folder, you can find the information regarding the segmentation part of the pipeline. So far different versions of [MaskRCNN](https://arxiv.org/pdf/1703.06870.pdf) and [UNet](https://arxiv.org/pdf/1505.04597.pdf) have been implemented. You can simply add more models by adding your own model and reference it in the code.

## Running Segmentation Tasks

Here is an example of running the Segmentation task:

An example of how to run a semantic segmentation or instance segmentation task can be found in the [examples](examples) folder.  
Therefore the config.json file from the example folder has to be moved into the Pipeline folder and executed using the main.
Folder names must not be changed.