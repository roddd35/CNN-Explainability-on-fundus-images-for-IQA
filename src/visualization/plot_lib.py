import itertools
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score 

'''
Functions to plot graphs on the model
performance.
call_epochs_plot aims to plot accuracy/
loss X epochs, without covering to many
code lines.
Other parameters plotted are confusion
matrix, pr curve and roc curve.
'''

'''
Call training by epochs
plot for loss and accuracy
'''
def call_epochs_plot(loss_hist, vloss_hist, acc_hist, vacc_hist, save_path):
    plot_metric_epochs(loss_hist, vloss_hist, "Training Loss", "Validation Loss", "Epochs", "Cross-Entropy Loss", "Loss Comparison", "loss_evolution", save_path)
    plot_metric_epochs(acc_hist, vacc_hist, "Training Accuracy", "Validation Accuracy", "Epochs", "Accuracy", "Accuracy Comparison", "accuracy_evolution", save_path)

'''
Actually plot the metrics
evolution by epoch
'''
def plot_metric_epochs(train_data, val_data, train_label, val_label, x_label, y_label, title, file_name, save_path):
    plt.plot(train_data, color='royalblue', label=train_label, linewidth=.85)
    plt.plot(val_data, color='orange', label=val_label, linewidth=.85)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(color='#EEEEEE', linewidth=0.7)
    plt.legend(loc='best')
    plt.savefig(save_path + file_name + ".png")
    plt.close()

'''
Plot confusion matrix
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão, sem normalização')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Label verdadeiro')
    plt.xlabel('Label previsto')

'''
Call the confusion matrix
plot function
'''
def call_cm_plot(y_true, Y_pred, save_path, cm):
    target_names = ['Inadequate', 'Adequate']

    plt.figure(figsize=(10, 7))
    plot_confusion_matrix(cm, classes=target_names, title='Confusion Matrix')
    plt.savefig(save_path + ".png")
    plt.clf()   # clean figure for the second plot

'''
Plot ROC Curve for 
AUC purposes
'''
def plot_roc_curve(y_true, Y_pred, save_path):
    # fpr_model, tpr_model, thresholds_model = roc_curve(y_true, Y_pred.ravel())
    fpr_model, tpr_model, thresholds_model = roc_curve(y_true, Y_pred[:, 1])
    auc_model = auc(fpr_model, tpr_model)

    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr_model, tpr_model, label='Model (area = {:.3f})'.format(auc_model))
    plt.grid(color='#EEEEEE', linewidth=0.7)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(save_path + ".png") 
    plt.clf()

'''
Plot Precision-Recall
curve
'''
def plot_pr_curve(y_true, Y_pred, save_path):
    # precision, recall, thresholds = precision_recall_curve(y_true, Y_pred.ravel())
    # average_precision = average_precision_score(y_true, Y_pred.ravel())
    precision, recall, thresholds = precision_recall_curve(y_true, Y_pred[:, 1])
    average_precision = average_precision_score(y_true, Y_pred[:, 1])

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label='Model (Avg Precision = {:.3f})'.format(average_precision))
    plt.grid(color='#EEEEEE', linewidth=0.7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
    plt.savefig(save_path + ".png")
    plt.show()
    plt.clf()