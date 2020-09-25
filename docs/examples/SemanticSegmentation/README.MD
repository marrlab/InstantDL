# Semantic Segmentation Example

In the following, we are trying to solve a lung segmentation task from CT images with the InstantDL.
Each image and its corrsponding groundtruth must have the same name and should be put in the train or test folder.
The jupyter-notebook called "CreateFolderstrucutre_and_SplitTrainTestset.ipynb" located in the docs/preprocessing folder might help you to set up this folder structure and crate a random train and test set split.

```
path
├── train
│   ├── image
│   │    ├── VESSEL12_01.mhd100.tif
│   │    ├── VESSEL12_01.mhd101.tif
│   │    ├── VESSEL12_01.mhd102.tif
│   │    ├── .
│   │    ├── .
│   │    ├── .
│   └── groundtruth
│   │    ├── VESSEL12_01.mhd100.tif
│   │    ├── VESSEL12_01.mhd101.tif
│   │    ├── VESSEL12_01.mhd102.tif
│   │    ├── .
│   │    ├── .
│   │    ├── .
└── test
   ├── image
   │    ├── VESSEL12_01.mhd111.tif
   │    ├── VESSEL12_01.mhd115.tif
   │    ├── VESSEL12_01.mhd120.tif
   │    ├── .
   │    ├── .
   │    ├── .
   └── groundtruth
   │    ├── VESSEL12_01.mhd111.tif
   │    ├── VESSEL12_01.mhd115.tif
   │    ├── VESSEL12_01.mhd120.tif
   │    ├── .
   │    ├── .
   │    ├── .
```
After that the data is provided in the desired shape, you can simply use the code with this small snippet:

```python
from instantdl import GetPipeLine

configs = { "use_algorithm": "SemanticSegmentation", 
            "path": "docs/examples/SemanticSegmentation/",
            "pretrained_weights": "docs/examples/SemanticSegmentation/logs/pretrained_weights_Lung_SemanticSegmentation.hdf5",
            "batchsize": 1,
            "iterations_over_dataset": 0,
            "data_gen_args": {
                     "save_augmented_images": False,
                     "resample_images": False,
                     "std_normalization": False,
                     "feature_scaling": False,
                     "horizontal_flip": False,
                     "vertical_flip": True,
                     "poission_noise": False,
                     "rotation_range": False,
                     "zoom_range": False,
                     "contrast_range": False,
                     "brightness_range": False,
                     "gamma_shift": False,
                     "threshold_background_image": False,
                     "threshold_background_groundtruth": False,
                     "binarize_mask": False
            },
            "loss_function": "binary_crossentropy",
            "num_classes": 1,
            "image_size": None,
            "calculate_uncertainty": False,
            "evaluation": True}

pipeline = GetPipeLine(configs)

pipeline.run()
```
The pipeline will start training on the images in the train folder and once its finished evaluate the trained model on the images in the test folder.
If the uncertainty calculation is desired it will evaluate 20 trained models to obtain an uncertainty score. This might take some time.
Once InstantDL is finished it will output as png files visualizing the result and .npy files containing the predicted masks to the results to the result folder.
The performance can be evaluated if evaluation is set to True, then InstantDL will save evaluations such as the mean squared error to the insights folder as a .txt file and print visualizations of the segmentation to the evaluation folder.

As you can see, it is very straightforward to use the pipeline and there is no need for more programming.