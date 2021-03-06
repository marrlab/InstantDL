# Evaluation

The evaluation is implemented in jupyter-notebooks.

## Quantitative and qualitative evaluation

For semantic segmentation, instance segmentation and regression the network predictions are first saved as image stacks into the "insights"-folder which is created to ease evaluation of large datasets with limited CPU capabilities. In a second step the predictions can be qualitatively and quantitatively evaluated.
The quantitative evaluation calculates the accuracy, mean relative and absolute error, Pearson correlation coefficient and Jaccard Index over all pixels of the test dataset and generates boxplots to visualize quantitative model performance.
The quantitative evaluation plots the input images side-by-side to the corresponding labels and predictions and plots an error map between the labels and predictions to visualize training performance.

## Classification evaluation

For Classification the labels predicted from the images in the testset are compared to the true labels and different error scores are calculated. A confusion matrix and a receiver operating characteristic (ROC) curve is visualized.