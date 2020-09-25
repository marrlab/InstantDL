# Instance Segmentation Example

In the following, we are trying to solve a nuclei detection regression problem with the pipeline. For using instance segmentation the groundtruth masks, which each contain one segmentation mask for one instance the folder setup is a little more compliated.
For each image one folder with the name of the image, but without the file ending) needs to be created in the train or test folder. In this folder an image and mask forlder is created containing the image and corresponding masks.
The jupyter-notebook called "PrepareDataforMaskRCNN.ipynb" located in the docs/preprocessing folder might help you to set up this folder structure if you already have sorted the images into a train and test folder (similar to the semantic segmentaiton setup).

```
path
├── train
│   ├── fc345dac2205deb169bd70197f07f053bada80b61ffa69fdfb490758323ead69
│   │   │    ├── image
│   │   │    │    ├── fc345dac2205deb169bd70197f07f053bada80b61ffa69fdfb490758323ead69.png
│   │   │    ├── mask
│   │   │    │    ├── 0e8ae61720424eb6445e537b3e7bb83d295e90a4b9afff5adaab9f8ef7323d96.png
│   │   │    │    ├── 1ebc390c94e61dec3a268c93625f4145e4663389591522141e54e85c91b4732a.png
│   │   │    │    ├── .
│   │   │    │    ├── .
│   │   │    │    ├── .
│   ├── fd8065bcb1afdbed19e028465d5d00cd2ecadc4558de05c6fa28bea3c817aa22
│   │   │    ├── image
│   │   │    │    ├── fd8065bcb1afdbed19e028465d5d00cd2ecadc4558de05c6fa28bea3c817aa22.png
│   │   │    ├── mask
│   │   │    │    ├── 2be95a388114cf8b99fd6e4c2b8c8a6ca3c8cf14ca83b4088638b28fca4b216e.png
│   │   │    │    ├── 3bd6ff5c04b482a9a5bc83260e786dde55252f76463fb0d106e65cbf84f94b5b.png
│   │   │    │    ├── .
│   │   │    │    ├── .
│   │   │    │    ├── .
│   ├── .
│   ├── .
│   ├── .
└── test
    ├── 4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40
    │   │    ├── image
    │   │    │    ├── 4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40.png
    │   │    ├── mask
    │   │    │    ├── 00c9db67855c83b7feac9f682e6246820a240e2204e5e487fa2b49ad08e2a547.png
    │   │    │    ├── 0d7cca22d70ea262c2e4d1cc66e04e1e438da1bdcd32e881cf57c016ef3ad8c2.png
    │   │    │    ├── .
    │   │    │    ├── .
    │   │    │    ├── .
    ├── 4ff152d76db095f75c664dd48e41e8c9953fd0e784535883916383165e28a08e
    │   │    ├── image
    │   │    │    ├── 4ff152d76db095f75c664dd48e41e8c9953fd0e784535883916383165e28a08e.png
    │   │    ├── mask
    │   │    │    ├── 7d7f8396b6f1e2a1f8417c3f42062a7dc0252fcec1a361bb85466197931a070d.png
    │   │    │    ├── 08ae7ecfa3f66f8ba283d7922af77248cddad8f01793d0f20e6ba459951db150.png
    │   │    │    ├── .
    │   │    │    ├── .
    │   │    │    ├── .
    ├── .
    ├── .
    ├── .
```


After that the data is provided in the desired shape, you can simply use the code with this small snippet:

```python
from instantdl import GetPipeLine

configs = {
            "use_algorithm" : "InstanceSegmentation", # mandatory field
            "path" : "docs/examples/InstanceSegmentation", # mandatory field
	        "pretrained_weights" : None,
            "batchsize" : 1,
            "iterations_over_dataset": 10,
            "data_gen_args": {
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
            "loss_function": "mse",
            "num_classes": 1,
            "image_size": None,
            "calculate_uncertainty": True,
            "evaluation": True}

pipeline = GetPipeLine(configs)

pipeline.run()
```
The pipeline will start training on the images in the train folder and once its finished evaluate the trained model on the images in the test folder.
For instance segmentation uncertainty estimation is not supported.
Once InstantDL is finished it will output as png files visualizing the result and .npy files containing the predicted masks to the results to the result folder.
The performance can be evaluated if evaluation is set to True, then InstantDL will save evaluations such as the mean squared error to the insights folder as a .txt file and print visualizations of the segmentation to the evaluation folder.

As you can see, it is very straightforward to use the pipeline and there is no need for more programming.

