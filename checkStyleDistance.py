from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

def load_image(img_path, size = 200, shape = None):
    image = Image.open(img_path).convert("RGB")

    # if max(image.size) > max_size:
    #     size = max_size
    # else:
    #     size = max(image.size)

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3, : , :].unsqueeze(0)
    return image

kinds = ["Drawing", "Engraving", "Oil", "Oriental", "Pastel", "Watercolor" ]
num_kinds = len(kinds)
num_style_layers = 5

def get_index_images(index):
    images = []
    for  kind in range(num_kinds):
        path = kinds[kind]  + '/'+ str(index) + '.jpg'
        images.append(load_image(path).to(device))
    return images


# oriental =load_image('Oriental/0.jpg').to(device)
# engraving = load_image('Engraving/0.jpg', shape = oriental.shape[-2:]).to(device)
# drawing = load_image('Drawing/0.jpg', shape = oriental.shape[-2:]).to(device)
# oil = load_image('Oil/0.jpg', shape = oriental.shape[-2:]).to(device)
# pastel = load_image('Pastel/0.jpg', shape = oriental.shape[-2:]).to(device)
# watercolor = load_image('Watercolor/0.jpg', shape = oriental.shape[-2:]).to(device)

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

# display the images
# ori = plt.subplot(2, 3, 1)
# e = plt.subplot(2, 3, 2)
# d = plt.subplot(2, 3, 3)
# oi = plt.subplot(2, 3, 4)
# p = plt.subplot(2, 3, 5)
# w = plt.subplot(2, 3, 6)
#
# ori.imshow(im_convert(oriental))
# e.imshow(im_convert(engraving))
# d.imshow(im_convert(drawing))
# oi.imshow(im_convert(oil))
# p.imshow(im_convert(pastel))
# w.imshow(im_convert(watercolor))



def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 0.1,
                 'conv2_1': 0.2,
                 'conv3_1': 0.3,
                 'conv4_1': 0.4,
                 'conv5_1': 0.5}

style_weight = 1e6  # beta

all_data_features = []
for idx in range(num_kinds):
    data_features = get_features(data_images[idx], vgg)
    all_data_features.append(data_features)

#test
for test_idx in range(num_kinds):
    test_features = get_features(test_images[test_idx], vgg)



    for data_idx in range(num_kinds):
        loss = 0

        data_features = all_data_features[data_idx]

        for layer in style_weights:


                test_feature = test_features[layer]
                test_gram = gram_matrix(test_feature)

                _, d, h, w = test_feature.shape

                data_feature = data_features[layer]
                data_gram = gram_matrix(data_feature)

                layer_loss = style_weights[layer] *  torch.mean((test_gram - data_gram)**2)
                loss += layer_loss / (d * h * w)

        print("Test: ", kinds[test_idx], " Data: ",kinds[data_idx], "loss: ", loss)



















