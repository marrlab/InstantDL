# Classification

In this folder, you can find the information regarding the classification part of the pipeline. So far different versions of [ResNet](https://arxiv.org/pdf/1512.03385.pdf) have been implemented. You can simply add more models by adding your own model and reference it in the code.
Dropout has been added to the ResNet50 to enable uncertainty quantification using Monte Carlo Dropout.

## Running Classification Tasks

An example of how to run a classification task can be found in the [examples](examples) folder. Therefore the config.json file from the example folder has to be moved into the Pipeline folder and the main script has be executed.
Folder names must not be changed.

### Semi-supervised learning (Pseudo labeling)
Semi-supervised learning can be enabled by adding the following attribute in config.json:
```json
"semi_supervised": true
```