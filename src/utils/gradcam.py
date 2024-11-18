import os
import sys
import cv2 # type: ignore
import numpy as np
import torch
import torch.nn as nn

sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/data')
sys.path.append('/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/src/models')

from train_model import Net # type: ignore
from import_data import import_gradcam_data # type: ignore

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
Definition of the Gradcam network
Load trained network and separate 
convolutional and classification
features from the model
'''

class Gradcam_Net(nn.Module):
    def __init__(self, weights_path):
        super(Gradcam_Net, self).__init__()

        # load pre-trained model
        self.model = Net()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))

        # separate the trained network into conv features and classifier
        self.features_conv = self.model.features[:30]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # self.classifier = self.model.classifier essa linha falha pois não usamos a camada classifier em Net()
        self.classifier = nn.Sequential(
            self.model.fc1,
            self.model.relu,
            self.model.dropout,
            self.model.fc2
        )
 
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

def apply_gradcam(weights_path):
    img_path = '/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/images/gradcam/all_images'
    gradcam_loader = import_gradcam_data(img_path)

    model = Gradcam_Net(weights_path)
    model = model.to(device)

    model.eval()

    # Calculate gradcam for a specific image
    img, _ = next(iter(gradcam_loader))
    img = img.to(device)

    pred = model(img)
    pred_class = pred.argmax(dim=1).item()
    pred[0, pred_class].backward()

    # get the gradients from the model
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations from the last conv layer
    activations = model.get_activations(img).detach()
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()

    # apply ReLU on top of the heatmap (L_{Grad-CAM}^C = ReLU(\sum_k A^k \times \alpha_c^k))
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)   # normalize
    heatmap = heatmap.numpy()

    # interpolate heatmap and project it onto original image
    img = cv2.imread(os.path.join(img_path, 'img/img04490.jpg'))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(os.path.join(img_path, 'img/img04490_grad.jpg'), superimposed_img)