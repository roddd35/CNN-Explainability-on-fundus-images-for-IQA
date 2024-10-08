import sys

sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/data')
sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/models')
sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/visualization')

from import_data import import_data
from train_model import train_model
from plot_lib import call_epochs_plot

'''
Main algorithm execution
import images with ImageFolder() pytorch method
call function to define and train model using ResNet
'''

def main():
    data_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos'
    # data_path = '/home/rodrigocm/scratch/datasets/eyeq/images'
    save_images_path = '/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/images/charts/'
    epochs = 30
    lr = 0.0001

    print("Importando imagens...")
    train_loader, val_loader, test_loader = import_data(data_path)

    train_features, train_labels = next(iter(train_loader))
    test_features, test_labels = next(iter(test_loader))
    val_features, val_labels = next(iter(val_loader))

    print(f"Concluído\nShape das imagens {train_features.size()}")

    model, loss_hist, vloss_hist, acc_hist, vacc_hist = train_model(train_loader, val_loader, epochs, lr)
    call_epochs_plot(loss_hist, vloss_hist, acc_hist, vacc_hist, save_images_path)

main()

# testar o modelo com o eyeq
# testar o modelo no test_loader
# testar outras metricas tambem (precision, recall, auc)
# plotar roc-auc, pr-curve, matriz de confusao
# arrumar o modelo para tentar alcancar a mesma acuracia que o tensorflow