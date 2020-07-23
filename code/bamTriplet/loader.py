import matplotlib.pyplot as plt
import numpy as np
import random
import os

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, models

from net import StyleNet, TripleNet

ROOT = '/Users/sumin/PycharmProjects/NeuralStyleTransfer/data/BAM'
MEDIA = ['3DGraphics', 'Comic', 'Oil', 'Pen', 'Pencil', 'VectorArt', 'Watercolor']
CONTENT = ['Bicycle', 'Bird', 'Cars', 'Cat', 'Dog', 'Flower', 'People', 'Tree']
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # torvision은 데이터셋 출력을 [0,1] 범위를  [-1, 1]의 범위로 변

EPOCH = 20
NUM_MEDIA = len(MEDIA)
NUM_CONTENT = len(CONTENT)
NUM_RELEVANCE = 2


class Loader():
    def __init__(self):
        self.loaders = []

    def forward(self):
        for m in range(NUM_MEDIA):
            dataset = torchvision.datasets.ImageFolder(root=os.path.join(ROOT, MEDIA[m]), transform=TRANSFORM)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
            for i, data in enumerate(dataloader):
                img = data[0]
                c = data[1].item()
                element = [m, c, img]
                self.loaders.append(element)

        return self.loaders

    def countRelevance(self, input1, input2):
        n = 0
        for i in range(NUM_RELEVANCE):
            if input1[i] == input2[i]:
                n += 1
        return n

    def getPosImg(self, a):
        img = []
        while (1):
            idx = random.randint(0, len(self.loaders) -1)
            p = self.loaders[idx]
            if self.countRelevance(a, p) >= int(NUM_RELEVANCE / 2):  # relevance가 절반이상 같으면 pos image
                img = self.loaders[idx][-1]
                break;
        return img # img는 언제나 self.loader 배열에 마지막에

    def getNegImg(self, a):
        img = []
        while (1):
            idx = random.randint(0, len(self.loaders)-1)
            n = self.loaders[idx]
            if self.countRelevance(a, n) < int(NUM_RELEVANCE / 2):  # relevance가 절반이상 같으면 pos image
                img = self.loaders[idx][-1]
                break;
        return img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataLoader = Loader()
data = dataLoader.forward()
tripleNet = net.TripleNet(net.StyleNet(), net.ContentNet())
tripleNet = tripleNet.to(device)

parameters = tripleNet.parameters()
parameters.requires_grad = True
optimizer = optim.Adam(parameters, lr=0.01)
optimizer.zero_grad()

for epoch in range(EPOCH):
    for idx in range(len(data)):
        a = data[idx]
        a_img = a[-1]
        p_img = dataLoader.getPosImg(a)
        n_img = dataLoader.getNegImg(a)

        anchor, positive, negative = tripleNet.forward(a_img, p_img, n_img)






















