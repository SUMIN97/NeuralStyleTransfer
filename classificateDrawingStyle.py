import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
classes = ['drawing','oil', 'oriental','illustration','watercolor']


#torchvision은 데이터셋을 불러올때 전처리 작업을 간단히 진행할 수 있는 lib

#torvision은 데이터셋 출력을 [0,1] 범위를 갖는 pilimage이므로 [-1, 1]의 범위로 변
transform = transforms.Compose(
    [   transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root = 'data/',  transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)

testset = torchvision.datasets.ImageFolder(root = 'test/',  transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding = 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding = 1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            # 64 112 64
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 128 56 32
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 256 28 16
            nn.Conv2d(256, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # 512 14 8
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1), nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )

        self.avg_pool = nn.AvgPool2d(7)
        #512, 1, 1
        self.classfier = nn.Linear(512, len(classes))

    def forward(self, x):
        print(x.size())
        features = self.conv(x)
        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)

        return x, features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net = net.to(device)
param = list(net.parameters())




criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(),lr=0.00001)

for epoch in range(200):
    running_loss = 0.0

    for i,data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #zero the parameter gradients
        optimizer.zero_grad()

        outputs, f = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %d] loss: %.3f' % epoch+1, i+1, running_loss/50)
            running_loss = 0.0
    print('Finisned Training')

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

#기록 추적 방
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # images = images.cuda()
        # labels = labels.cuda()
        outputs, _ = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(64):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))











