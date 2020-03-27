# Deep Learning Pipeline

Our pipeline enables non-experts to use state-of-the art deep learning methods on biomedical image data. In order to reduce potential point of errors in an experimental setup we have highly automated and standardized as well as debugged and tested our pipeline. Without any parameter tuning we have benchmarked it on XX datasets. For customization of the pipeline to specific tasks all code is easily accessibility. 

<img src="documentation/Figure1.jpeg" alt="architecture" width="700" class="center"/>

## Folders

Here you can find the information about the folders in the code:

- [classification](classification)
- [data_generator](data_generator)
- [Preprocessing_Evaluation](Preprocessing_Evaluation)
- [segmentation](segmentation)

## Dependencies

For running the code, you need to have Python 3.7 or higher installed. In addition, these are the main dependencies:

```json
{
   "cudatoolkit":"10.1.243",
   "cudnn":"7.6.5",
   "h5py":"2.9.0",
   "hdf5":"1.10.4",
   "imageio":"2.6.1",
   "keras":"2.2.4",
   "matplotlib":"3.1.1",
   "numpy":"1.16.4",
   "python":"3.6.7",
   "scikit-image":"0.15.0",
   "scikit-learn":"0.21.3",
   "scipy":"1.3.0",
   "tensorboard":"1.14.0",
   "tensorflow":"1.14.0",
   "tensorflow-gpu":"1.14.0"
}
```

## How to use the code

```bash
The pipeline is based on folders. Please put your data manually in the corresponding folders as illustrated by the figure above. The folder names must not be changed as this will stop the pipeline from functioning.

The config.json file in the config folder must be used to set parameters for the pipeline. After this the pipeline is started by executing the main.py file. Predictins from the testset will be saved after training to the Results folder which will automatically be created. 

From there evaluations using jupyter-notebooks from the Evaluation folder can be used for visual and statistical assessment. Therefore only the path in the jupyter-notebook files has to be adapted. 

Possible setting in the config.json file are: 
"use_algorithm": "Regression", "SemanticSegmentation", "Instance Segmentation" or "Classification"
"path": "data/Ouncomol_NucEnvelope_3Dnew", #Set the path to your project directory here
"use_pretrained_weights": false, # Set to true if you want to use pretrained weights
"pretrained_weights_path": false, # Set a relative file path from your project directory with the filename here. 
"batchsize": 2, # Set the batchsize depeding on your GPU capabilities
"Iterations_Over_Dataset": 200, # Set how many iterations over the dataset should be taken for learning. It might stop automatically if no improvement on the validation set was measured after 25 epochs

Set data augmentation parameters here
"save_augmented_images": false, # true or false
"resample_images": false, # true or false
"std_normalization": false, # true or false
"feature_scaling": false, # true or false
"horizontal_flip": false, # true or false
"vertical_flip": false, # true or false
"poission_noise": false, # false or float 
"rotation_range": false, # false or float (degrees)
"zoom_range": false, # false or float (magnification)
"contrast_range": false, # false or float 
"brightness_range": false, # false or float 
"gamma_shift": false, # false or float (gamma shift parameter)
"threshold_background_image": false, # true or false
"threshold_background_groundtruth": false, # true or false
"gaussian_blur_image": false, # true or float
"gaussian_blur_label": false, # true or  # true or false
"binarize_mask": false # true or false

Set parameters for training here. Number of classes should be 1 for binary segmenation tasks
"loss_function": "mse",
"num_classes": 1, # Number of classes should be 1 for binary segmenation tasks
"Image_size": null, # null or tuple with dimensions of desired image size in format (x-dim, y-dim, [z-dim,] channels)
"calculate_uncertainty": false # true or false


python main.py --config ./configs/config.json
```

## Run examples:
One example for each task of Semantic segmentation, Instance segmentation, regression and classification is in the examples folder. 

To run them only the corresponding config.json file from the example folder must be copied into the configs folder and renamed to "config.json". 

Then the main.py can be executed and the example will run with the provided data. 

Please don't expect to achieve competitive results on these datasets, as they are very small and only for illustration purposes. 


## Comparison

| Application | InstantDL | Open ML  | Google Cloud AI  | ImJoy |
| ------ | ------ | ------  | ------  | ------ |
| Host | Local, cluster or in Google-Colab | Web based  | Web based  | Web based |
| Data privacy | Yes | Shared with upload | Yes  | Yes |
| Target audience | Biomedical researchers  | Researchers and Developers  | Enterprises  | Researchers and Developers |
| Developed for  | Biomedical images | All kinds of data  | All kinds of data  | All kinds of data |
| Customizability of Code | Code fully accessible | Code fully accessible  | Code not available  | Possible through writing a plugin |
| Debugged and tested | Yes | Yes  | Yes  | Yes |
| Cost | Free | Free  | Free  | Payment plan |































## TODO

- [x] classifciation
- [ ] making the code modular
- [ ] adding documentation
- [ ] adding example
- [ ] add docker
- [ ] add objects instead of the main functions
- [ ] add tests

