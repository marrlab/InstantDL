In the following, we are trying to solve a in-silico staining from brightfield images to nuclear envelope staining regression problem with the pipeline.
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

The groundtruth images should have the same name as their image counterparts. These can be manually sorted into the corresponding folders.

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

As you can see, it is very straightforward to use the pipeline and there is no need for more programming.

