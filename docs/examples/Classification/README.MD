In the following, we are trying to solve a simple MNIST classification problem with the pipeline. For simplification, we only solve a binary classification for distinguishing between `1` and `9`. First, the dataset and the groundtruth should be provided. The data folder should look like this:

```
path
├── train                    
│   ├── image
│   │    ├── 000003-num1.png
│   │    ├── 000004-num9.png
│   │    ├── 000006-num1.png
│   │    ├── .
│   │    ├── .
│   │    ├── .
│   │    └── 059994-num1.png     
│   └── groundtruth  
│        └── groundtruth.csv
│
└── test                    
   ├── image
   │    ├── 000002-num1.png
   │    ├── 000005-num1.png
   │    ├── 000007-num9.png
   │    ├── .
   │    ├── .
   │    ├── .
   │    └── 009994-num1.png     
   └── groundtruth  
        └── groundtruth.csv
```

And the `groundtruth.csv` should look like this:

| filename        |   groundtruth   |
| ------          |    ------       |
| 015312-num9.png	|        1        | 
| 012725-num1.png	|        0        |
|      . 	      |         .       |
|      . 	      |         .       |
|      . 	      |         .       |

After that the data is provided in the desired shape, you can simply use the code with this small snippet:

```python
from instantdl import GetPipeLine

pipeline = GetPipeLine( use_algorithm = "Classification",	
                        path= "docs/examples/Classification",
	                     pretrained_weights= "docs/examples/data/Classification/logs/pretrained_weights_Classification.hdf5",
                        batchsize= 2,
                        iterations_over_dataset= 0,
                        data_gen_args= {
                              "save_augmented_images": False,
                              "resample_images": False,
                              "std_normalization": False,
                              "feature_scaling": False,
                              "horizontal_flip": False,
                              "vertical_flip": False,
                              "poission_noise": 1,
                              "rotation_range": 20,
                              "zoom_range": False,
                              "contrast_range": 1,
                              "brightness_range": 1,
                              "gamma_shift": 0,
                              "threshold_background_image": False,
                              "threshold_background_groundtruth": False,
                              "binarize_mask": False
                        },
                        loss_function= "binary_crossentropy",
                        num_classes= 2,
                        image_size= None,
                        calculate_uncertainty= True,
                        evaluation= True
)

pipeline.run()
```
The pipeline will start training on the images in the train folder and once its finished evaluate the trained model on the images in the test folder.
If the uncertainty calculation is desired it will evaluate 20 trained models to obtain an uncertainty score. This might take some time. Once InstantDL is finished it will output a .csv file containing the results to the result folder.
The performance can be evaluated if evaluation is set to True, then InstantDL will save evaluations such as the area under curve and a confusion matrix as well as common error metrics to the insights folder.

As you can see, it is very straightforward to use the pipeline and there is no need for more programming.

