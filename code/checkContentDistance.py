from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models

vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
vgg.to(device)


def load_image(img_path, size = 600, shape = None):
    image = Image.open(img_path).convert("RGB")

    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, : , :].unsqueeze(0)
    return image

kinds = ["Bicycle", "Bird", "Building", "Car", "Cat", "Dog", "Flower", "People", "Tree" ]
num_kinds = len(kinds)

def get_index_images(index):
    images = []
    for  kind in range(num_kinds):
        path = "/home/lab/Documents/SWMaestro/NeuralStyleTransfer/bam/diffContent/3DGraphics/"+kinds[kind]  + '/'+ str(index) + '.jpg'
        images.append(load_image(path).to(device))
    return images



data_images = get_index_images(0)
test_images = get_index_images(1)



# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'21': 'conv4_2'} ## content representation

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

content_weight = 1

all_data_features = []
for idx in range(num_kinds):
    data_features = get_features(data_images[idx], vgg)
    all_data_features.append(data_features)

#test
for test_idx in range(num_kinds):
    test_features = get_features(test_images[test_idx], vgg)

    for data_idx in range(num_kinds):
        data_features = all_data_features[data_idx]
        content_loss = torch.mean((test_features['conv4_2'] - data_features['conv4_2']) ** 2)

        print("Test: ", kinds[test_idx], " Data: ",kinds[data_idx], "loss: ", content_loss)



















