import sys

sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/data')
sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/utils')
sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/models')
sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/visualization')

from gradcam import apply_gradcam # type: ignore
from import_data import import_data # type: ignore
from test_model import test_model # type: ignore
from train_model import train_model # type: ignore
from plot_lib import call_epochs_plot, call_cm_plot, plot_roc_curve, plot_pr_curve # type: ignore

'''
Main algorithm execution
import images with ImageFolder() pytorch method
call function to define and train model using ResNet
'''

def main():
    data_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos'
    # data_path = '/home/rodrigocm/scratch/datasets/eyeq/images'
    save_images_path = '/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/images/charts/'
    weights_path = '/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/model/brset_model.pth'

    # define parameters
    epochs = 30
    lr = 0.00001

    # import dataset
    print("Importando imagens...")
    train_loader, val_loader, test_loader = import_data(data_path)

    train_features, _ = next(iter(train_loader))

    print(f"Concluído\nShape das imagens {train_features.size()}")

    # train dataset
    loss_hist, vloss_hist, acc_hist, vacc_hist = train_model(train_loader, val_loader, epochs, lr)
    call_epochs_plot(loss_hist, vloss_hist, acc_hist, vacc_hist, save_images_path)

    # evaluate model
    # ajustar o threshold aqui na hora de gerar o report
    y_true, y_pred, y_pred_prob, cm = test_model(test_loader, weights_path)

    # plot cm, roc and pr curve
    call_cm_plot(y_true, y_pred, save_path="/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/images/charts/confusion_matrix", cm=cm)
    plot_roc_curve(y_true, y_pred_prob, save_path="/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/images/charts/roc_curve")
    plot_pr_curve(y_true, y_pred_prob, save_path="/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/images/charts/pr_curve")

    apply_gradcam(weights_path)

main()
