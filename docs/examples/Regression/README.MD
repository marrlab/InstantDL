In the following, we are trying to solve a in-silico staining from brightfield images to nuclear envelope staining regression problem with the pipeline.
Each image and its corrsponding groundtruth must have the same name and should be put in the train or test folder.
The jupyter-notebook called "CreateFolderstrucutre_and_SplitTrainTestset.ipynb" located in the docs/preprocessing folder might help you to set up this folder structure and crate a random train and test set split.
```
path
├── train
│   ├── image
│   │    ├── 3500000876_100X_20170508_2-Scene-06-P16-E05.czi_depth_channel061630.npy
│   │    ├── 3500000876_100X_20170508_2-Scene-06-P16-E05.czi_depth_channel308030.npy
│   │    ├── 3500000876_100X_20170508_2-Scene-06-P16-E05.czi_depth_channel30830830.npy
│   │    ├── .
│   │    ├── .
│   │    ├── .
│   └── groundtruth
│   │    ├── 3500000876_100X_20170508_2-Scene-06-P16-E05.czi_depth_channel061630.npy
│   │    ├── 3500000876_100X_20170508_2-Scene-06-P16-E05.czi_depth_channel308030.npy
│   │    ├── 3500000876_100X_20170508_2-Scene-06-P16-E05.czi_depth_channel30830830.npy
│   │    ├── .
│   │    ├── .
│   │    ├── .
└── test
   ├── image
   │    ├── 3500000915_100X_20170519_1-Scene-6-P58-E05.czi_depth_channel0030.npy
   │    ├── 3500000915_100X_20170519_1-Scene-6-P58-E05.czi_depth_channel030830.npy
   │    ├── 3500000915_100X_20170519_1-Scene-6-P58-E05.czi_depth_channel030830.npy
   │    ├── .
   │    ├── .
   │    ├── .
   └── groundtruth
   │    ├── 3500000915_100X_20170519_1-Scene-6-P58-E05.czi_depth_channel0030.npy
   │    ├── 3500000915_100X_20170519_1-Scene-6-P58-E05.czi_depth_channel030830.npy
   │    ├── 3500000915_100X_20170519_1-Scene-6-P58-E05.czi_depth_channel030830.npy   │    ├── .
   │    ├── .
   │    ├── .
```

After that the data is provided in the desired shape, you can simply use the code with this small snippet:

```python
from instantdl import GetPipeLine

pipeline = GetPipeLine( use_algorithm = "Regression",
                        path= "docs/examples/Regression",
	                     pretrained_weights= None,
                        batchsize= 1,
                        iterations_over_dataset= 10,
                        data_gen_args= {
                              "save_augmented_images": False,
                              "resample_images": False,
                              "std_normalization": False,
                              "feature_scaling": False,
                              "horizontal_flip": True,
                              "vertical_flip": True,
                              "poission_noise": False,
                              "rotation_range": 20,
                              "zoom_range": 2,
                              "contrast_range": False,
                              "brightness_range": False,
                              "gamma_shift": False,
                              "threshold_background_image": False,
                              "threshold_background_groundtruth": False,
                              "binarize_mask": False
                        },
                        loss_function= "mse",
                        num_classes= 1,
                        image_size= None,
                        calculate_uncertainty= True,
                        evaluation= True
)

pipeline.run()
```
The pipeline will start training on the images in the train folder and once its finished evaluate the trained model on the images in the test folder.
If the uncertainty calculation is desired it will evaluate 20 trained models to obtain an uncertainty score. This might take some time.
Once InstantDL is finished it will output as png files visualizing the result and .npy files containing the predicted masks to the results to the result folder.
The performance can be evaluated if evaluation is set to True, then InstantDL will save evaluations such as the mean squared error to the insights folder as a .txt file and print visualizations of the regression to the evaluation folder.

As you can see, it is very straightforward to use the pipeline and there is no need for more programming.

