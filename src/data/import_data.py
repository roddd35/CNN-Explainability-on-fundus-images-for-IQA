import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torchvision.utils import save_image

# https://www.kaggle.com/code/gxkok21/resnet50-with-pytorch
# interessante para dar load nas imagens com class

# denormalize image to save
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    return tensor * std + mean

# save image to directory
def save_img_to_dir(img, label):
    denorm_image = denormalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    denorm_image = torch.clamp(denorm_image, 0, 1)

    print(f"Label: {label}")

    save_image(denorm_image, '/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/test.png')

# define transformations for training images
def set_train_transforms():
    train_transforms = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.RandomResizedCrop(size=(224, 224), antialias=True), # faz crops aleatorios, bom para generalizacao
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms

# define transformations for test and validation images
def set_test_transforms():
    test_transforms = v2.Compose([
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  

    return test_transforms

def load_dataset(data_path, dir, transforms):
    return datasets.ImageFolder(os.path.join(data_path, dir), transform=transforms)

def import_data(data_path):
    train_transforms = set_train_transforms()
    test_transforms = set_test_transforms()
    
    # load dataset with inferred labels and apply transformations
    train_data = load_dataset(data_path, 'train', train_transforms)
    validation_data = load_dataset(data_path, 'validation', test_transforms)
    test_data = load_dataset(data_path, 'test', test_transforms)

    # convert images loaded to data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size=128, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=12, pin_memory=True)

    print(f"Tamanho do conjunto de treino: {len(train_data)}")
    print(f"Tamanho do conjunto de validação: {len(validation_data)}")
    print(f"Tamanho do conjunto de teste: {len(test_data)}")

    # iterate through images and save a desirable one
    # images, labels = next(iter(test_loader))

    # save_img_to_dir(images[0], labels[0])

    return train_loader, val_loader, test_loader