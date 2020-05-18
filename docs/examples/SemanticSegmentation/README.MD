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
```
pipeline = GetPipeLine( "use_algorithm": "SemanticSegmentation",
                        "path": "docs/examples/SemanticSegmentation/",
                        "pretrained_weights_path": "docs/examples/SemanticSegmentation/logs/pretrained_weights_Lung_SemanticSegmentation.hdf5",
                        "batchsize": 1,
                        "iterations_over_dataset": 0,
                        "data_gen_args": {
                                "save_augmented_images": false,
                                        "resample_images": false,
                                        "std_normalization": false,
                                        "feature_scaling": false,
                                        "horizontal_flip": false,
                                        "vertical_flip": true,
                                        "poission_noise": false,
                                        "rotation_range": false,
                                        "zoom_range": false,
                                        "contrast_range": false,
                                        "brightness_range": false,
                                        "gamma_shift": false,
                                        "threshold_background_image": false,
                                        "threshold_background_groundtruth": false,
                                        "binarize_mask": false
                        },
                        "loss_function": "binary_crossentropy",
                        "num_classes": 1,
                        "image_size": null,
                        "calculate_uncertainty": false,
                        "evaluation": true
)

pipeline.run()
```
The pipeline will start training on the images in the train folder and once its finished evaluate the trained model on the images in the test folder.
If the uncertainty calculation is desired it will evaluate 20 trained models to obtain an uncertainty score. This might take some time.
Once InstantDL is finished it will output as png files visualizing the result and .npy files containing the predicted masks to the results to the result folder.
The performance can be evaluated if evaluation is set to True, then InstantDL will save evaluations such as the mean squared error to the insights folder as a .txt file and print visualizations of the segmentation to the evaluation folder.

As you can see, it is very straightforward to use the pipeline and there is no need for more programming.