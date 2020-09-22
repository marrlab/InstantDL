import numpy as np
import os
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
import logging

def get_auc(path, y_test, y_score, n_classes):
    '''
    calculates the area uncer curve and saves the AUC-curve to the insights folder
    :param path: path to directory
    :param y_test: groundtruth labels
    :param y_score: prediction labels
    :param n_classes: number of classes in classification task
    :return: Save figure to insights folder and return roc_auc values
    '''
    fpr = []
    tpr = []
    roc_auc = []
    for i in range(n_classes):
        fpr_out, tpr_out, _ = roc_curve(y_test[:, i].astype("float"), y_score[:, i].astype("float"))
        fpr.append(fpr_out)
        tpr.append(tpr_out)
        roc_auc.append(auc(fpr_out, tpr_out))
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label= "Class:" + str(i) +' ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='bottom right')
    plt.savefig(path + "/insights/AUC.png")
    return roc_auc

def get_confusion_matrix(path, Groundtruth, Results):
    '''
    :param path: path to project directory
    :param Groundtruth: groundtruth class labels
    :param Results: predicted class labels
    :return: safe confusion matrix to insight folder
    '''
    confusion_matr = confusion_matrix(Groundtruth, Results)
    normalized_confustion_matrix = confusion_matr.astype('float') / (confusion_matr.sum(axis=1)[:, np.newaxis]+10e-10)
#    logging.info("The confusion matrix is: \n", confusion_matrix)
    cmap = plt.cm.Blues
    title = 'Confusion matrix'
    fig, ax = plt.subplots()
    im = plt.imshow(normalized_confustion_matrix * 100, interpolation='nearest', cmap='binary')
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=15)
    plt.xticks(np.arange(confusion_matr.shape[0]), fontsize=15, rotation=90)
    plt.yticks(np.arange(confusion_matr.shape[1]), fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylabel('True Class', fontsize=15)
    plt.xlabel('Predicted Class', fontsize=15)
    plt.tight_layout()
    plt.savefig(path + "/insights/Confusion_Matrix.png") #TODO Plot cuts legend when saving

def load_data(path):
    '''
    imports the predictions and groundtruth and creates a table called Predictions_and_Results_Table in the insights folder
    containing the predictions and results for each filename
    :param path: path to project directory
    :return: the imported groundtruth, results and sigmoid outputs as lists
    '''
    Groundtruth_in = pd.read_csv(path + "/test/groundtruth/groundtruth.csv")
    Results_in = pd.read_csv(path + "/results/results.csv")
    Results = []
    Groundtruth = []
    Sigmoid_output = []
    Sigmoid_max_output = []
    os.makedirs(path + '/insights/', exist_ok = True)
    with open(path + '/insights/Predictions_and_Results_Table.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(['filename', 'prediction', 'groundtruth'])
        for i in range(len(Results_in)):
            loc = Groundtruth_in.loc[Groundtruth_in['filename'] == Results_in["filename"][i]]
            if str(loc['filename'].values)[2:-2] == Results_in["filename"][i]:
                Results.append(Results_in["prediction"][i])
                Groundtruth.append((loc["groundtruth"]).values.astype(int))
                Sigmoid_output_i = Results_in["Probability for each possible outcome"][i]
                Sigmoid_max_output.append(max(Sigmoid_output_i))
                Sigmoid_output_i = np.array((Sigmoid_output_i)[1:-1].split())
                Sigmoid_output.append(Sigmoid_output_i)
                writer.writerow([Results_in["filename"][i], Results_in["prediction"][i], (loc["groundtruth"].values)])
    Groundtruth = np.array(Groundtruth)
    Results = np.array(Results)
    Sigmoid_output = np.array(Sigmoid_output)
    return Groundtruth, Results, Sigmoid_output

def classification_evaluation(path):
    '''
    opens the results and groundtruth and caluclates the accuracy and area under curve

    :param path: path to the project directory
    '''
    Groundtruth, Results, Sigmoid_output = load_data(path)
    accuracy = accuracy_score(Groundtruth, Results)
    Sigmoid_output = np.squeeze(Sigmoid_output)
    logging.info("The accuracy score is: %f" % accuracy)
    n_classes = len(np.unique(Groundtruth))
    GT = to_categorical(Groundtruth)
    os.makedirs((path + "/insights/"), exist_ok=True)
    roc_auc = get_auc(path, GT, Sigmoid_output, int(n_classes))
    weigted_roc_auc = 0
    for index, i in enumerate(np.unique(Groundtruth)):
        logging.info("Groundtruth.count(i) %d %d" % (i, np.count_nonzero(Groundtruth == i)))
        weigted_roc_auc += np.count_nonzero(Groundtruth == i)*roc_auc[index]
    weigted_roc_auc /= len(Groundtruth)
    logging.info("The weigted roc auc is: %f" % weigted_roc_auc)
    get_confusion_matrix(path, Groundtruth, Results)
    f = open(path + '/insights/Evaluation_results.txt', 'a')
    f.write('\n')
    f.write('\n' + "Evaluated at: " + str(datetime.datetime.now())[:16])
    f.write('\n' + "Accuracy : " + str(accuracy))
    for i in range(len(roc_auc)):
        f.write('\n' + "Area under curve class " + str(i) + ": " + str(roc_auc[i]))
    f.close()