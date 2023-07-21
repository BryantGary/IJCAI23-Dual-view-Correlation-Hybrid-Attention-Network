# coding:utf-8
from __future__ import print_function
import numpy as np
#import mydataset
from torch.utils.data import Dataset
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
import os
from torchvision import models
from image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pretrain_CNN
import data_preprocess
from non_local_dot_product import NONLocalBlock1D
from correlation import corr
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='PyTorch Resnet Training')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
args = parser.parse_args()

def adjust_learning_rate(optimizer, decay_rate=.5):  
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    
    print("changing lr rate")
        
print('==> Preparing data..')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.groups= groups
        if (groups!=1):
          groups = groups-1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.planes= planes
        self.inplanes= inplanes
        self.stride = stride
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False) 
        self.downsample = downsample

        
    def forward(self, x):
        identity = x
    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
       # print("out",out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
               
        out += identity
        out = self.relu(out)
               
        return out


class MyDataset(Dataset):
    def __init__(self, imagesCC,labelsCC,imagesMLO,labelsMLO):
        self.imagesCC = imagesCC
        self.labelsCC = labelsCC
        self.imagesMLO = imagesMLO
        self.labelsMLO = labelsMLO

    def __getitem__(self, index):

        imgCC = torch.Tensor(self.imagesCC[index])
        targetCC =  self.labelsCC[index]
        imgMLO = torch.Tensor(self.imagesMLO[index])
        targetMLO =  self.labelsMLO[index]
        
        return imgCC,targetCC,imgMLO,targetMLO

    def __len__(self):
        return len(self.imagesCC)

# input image dimensions
img_rows, img_cols = 256, 256
img_channels = 1
# the data, shuffled and split between train and test sets
trXMLO, y_trainMLO, teXMLO, y_testMLO, trXCC, y_trainCC, teXCC, y_testCC = data_preprocess.loaddata()
trYMLO = y_trainMLO.reshape((y_trainMLO.shape[0],1))
teYMLO = y_testMLO.reshape((y_testMLO.shape[0],1))
trYCC = y_trainCC.reshape((y_trainCC.shape[0],1))
teYCC = y_testCC.reshape((y_testCC.shape[0],1))

ratio= trYMLO.sum()*1./trYMLO.shape[0]*1.

train_len = len(trXMLO)
test_len = len(teXMLO)
print('tr ratio'+str(ratio))
weights = np.array((ratio,1-ratio))
weights = torch.Tensor(weights)
weights = weights.cuda()

X_trainMLO = trXMLO.reshape(-1, img_channels, img_rows, img_cols)
X_testMLO = teXMLO.reshape(-1, img_channels, img_rows, img_cols)
X_trainCC = trXCC.reshape(-1, img_channels, img_rows, img_cols)
X_testCC = teXCC.reshape(-1, img_channels, img_rows, img_cols)

print('X_train shape:', X_trainMLO.shape)
print(X_trainMLO.shape[0], 'train samples')
print(X_testMLO.shape[0], 'test samples')

'''1ch->3ch'''
X_train_extendMLO = np.zeros((X_trainMLO.shape[0],3, 256, 256))
for i in range(X_trainMLO.shape[0]):
    rexMLO = np.resize(X_trainMLO[i,:,:,:], (256, 256))
    X_train_extendMLO[i,0,:,:] = rexMLO
    X_train_extendMLO[i,1,:,:] = rexMLO
    X_train_extendMLO[i,2,:,:] = rexMLO
X_trainMLO = X_train_extendMLO
X_test_extendMLO = np.zeros((X_testMLO.shape[0], 3 ,256, 256))
for i in range(X_testMLO.shape[0]):
    rexMLO = np.resize(X_testMLO[i,:,:,:], (256, 256))
    X_test_extendMLO[i,0,:,:] = rexMLO
    X_test_extendMLO[i,1,:,:] = rexMLO
    X_test_extendMLO[i,2,:,:] = rexMLO
X_testMLO = X_test_extendMLO

X_train_extendCC = np.zeros((X_trainCC.shape[0],3, 256, 256))
for i in range(X_trainCC.shape[0]):
    rexCC = np.resize(X_trainCC[i,:,:,:], (256, 256))
    X_train_extendCC[i,0,:,:] = rexCC
    X_train_extendCC[i,1,:,:] = rexCC
    X_train_extendCC[i,2,:,:] = rexCC
X_trainCC = X_train_extendCC
X_test_extendCC = np.zeros((X_testCC.shape[0], 3,256, 256))
for i in range(X_testCC.shape[0]):
    rexCC = np.resize(X_testCC[i,:,:,:], (256, 256))
    X_test_extendCC[i,0,:,:] = rexCC
    X_test_extendCC[i,1,:,:] = rexCC
    X_test_extendCC[i,2,:,:] = rexCC
X_testCC = X_test_extendCC

X_trainMLO = X_trainMLO.astype('float32')
X_testMLO = X_testMLO.astype('float32')  
X_trainCC = X_trainCC.astype('float32')
X_testCC = X_testCC.astype('float32')
                     
trainset =  MyDataset(X_trainCC,y_trainCC,X_trainMLO,y_trainMLO)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1)
testset =   MyDataset(X_testCC, y_testCC,X_testMLO, y_testMLO)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)


print('==> Building model..')

'''Loading pretrained model on ImageNet'''
resnet101 = models.resnet101(pretrained=True)
net_backbone = pretrain_CNN.CNN(Bottleneck,[3,4,23,3])
pretrained_dict = resnet101.state_dict()
model_dict = net_backbone.state_dict()
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net_backbone.load_state_dict(model_dict)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model_backbone = net_backbone
    def forward(self, CC,MLO):
        CC_predict, CC_feature = self.model_backbone(CC)
        MLO_predict, MLO_feature = self.model_backbone(MLO)
        corr_total = 0
        for i in range(CC_feature.size(3)):
             corr_total += corr(CC_feature[:,:,:,i], MLO_feature[:,:,:,i])
        
        correlation = corr_total
        
        return CC_predict,MLO_predict,correlation

net = Model()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

Loss_list = []
Accuracy_list = []

start_epoch=0
best_accCC=0
best_accMLO=0
best_accAVG = 0
best_aucCC=0
best_aucMLO=0
best_aucAVG = 0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_lossCC = 0
    train_lossMLO = 0
    train_losscorr = 0
    train_losstotal = 0 
    correctCC = 0
    totalCC = 0
    correctMLO = 0
    totalMLO = 0    
    phase_predCC = np.array([])
    phase_labelCC = np.array([])
    phase_predMLO = np.array([])
    phase_labelMLO = np.array([])
    
    for batch_idx, (inputsCC, targetsCC,inputsMLO, targetsMLO) in enumerate(trainloader):
        inputsCC, targetsCC = inputsCC.to(device), targetsCC.to(device)
        inputsMLO, targetsMLO = inputsMLO.to(device), targetsMLO.to(device)

        optimizer.zero_grad()
        outputsCC,outputsMLO,correlation = net(inputsCC,inputsMLO)
        
        lossCC = criterion2(outputsCC, targetsCC.long())
        lossMLO = criterion2(outputsMLO, targetsMLO.long())
        losscorr = correlation
        losstotal = lossCC + lossMLO - losscorr
        losstotal.backward(retain_graph=True)
        optimizer.step()

        train_lossCC += lossCC.item()
        train_lossMLO += lossMLO.item()
        train_losscorr += losscorr.item()
        train_losstotal += losstotal.item()
        _, predictedCC = outputsCC.max(1)
        _, predictedMLO = outputsMLO.max(1)

        totalCC += targetsCC.size(0)
        correctCC += predictedCC.eq(targetsCC.long()).sum().item()
        totalMLO += targetsMLO.size(0)
        correctMLO += predictedMLO.eq(targetsMLO.long()).sum().item()

        phase_predCC = np.append(phase_predCC, predictedCC.cpu().numpy())
        phase_labelCC = np.append(phase_labelCC, targetsCC.data.cpu().numpy())

        phase_predMLO = np.append(phase_predMLO, predictedMLO.cpu().numpy())
        phase_labelMLO = np.append(phase_labelMLO, targetsMLO.data.cpu().numpy())

    epoch_aucCC = roc_auc_score(phase_labelCC, phase_predCC)
    epoch_aucMLO = roc_auc_score(phase_labelMLO, phase_predMLO)

    print ('trainLossCC: %.3f, trainAccuCC: %.3f%% (%d/%d)' % (train_lossCC/(batch_idx+1), 100.*correctCC/totalCC, correctCC, totalCC))
    print ('trainLossMLO: %.3f, trainAccuMLO: %.3f%% (%d/%d)' % (train_lossMLO/(batch_idx+1), 100.*correctMLO/totalMLO, correctMLO, totalMLO)) 
    print ('trainepoch_aucCC: %.3f,trainepoch_aucMLO: %.3f' % (epoch_aucCC,epoch_aucMLO))
    print ('trainLosscorr: %.3f,trainLosstotal: %.3f' % (train_losscorr/(batch_idx+1), train_losstotal/(batch_idx+1)))
    print ('trainLosstotal: %.3f' % (train_losstotal/(batch_idx+1)))        
           
def test(epoch):
    global best_accCC
    global best_accMLO
    global best_accAVG
    global best_aucCC
    global best_aucMLO
    global best_aucAVG
    print ("best_accCC:",best_accCC, "best_accMLO:",best_accMLO)
    print ("best_accAVG:",best_accAVG)
    print ("best_aucCC:",best_aucCC,"best_aucMLO:",best_aucMLO)
    print ("best_aucAVG:",best_aucAVG)
    
    net.eval()
    test_lossCC = 0
    test_lossMLO = 0
    test_losscorr = 0
    test_losstotal = 0
    correctCC = 0
    correctMLO = 0
    totalCC = 0
    totalMLO = 0
    phase_predCC = np.array([])
    phase_labelCC = np.array([])
    phase_predMLO = np.array([])
    phase_labelMLO = np.array([])
    
    with torch.no_grad():
      for batch_idx, (inputsCC, targetsCC, inputsMLO, targetsMLO) in enumerate(testloader):
        inputsCC, targetsCC = inputsCC.to(device), targetsCC.to(device)
        inputsMLO, targetsMLO = inputsMLO.to(device), targetsMLO.to(device)#.long()


        outputsCC,outputsMLO, correlation = net(inputsCC,inputsMLO)#
        #print("test",outputs)
        lossCC = criterion1(outputsCC, targetsCC.long())
        lossMLO = criterion1(outputsMLO, targetsMLO.long())
        losstotal = lossCC + lossMLO 
        test_lossCC += lossCC.item()
        test_lossMLO += lossMLO.item()
        test_losstotal += losstotal.item()
        _, predictedCC = outputsCC.max(1)
        _, predictedMLO = outputsMLO.max(1)

        totalCC += targetsCC.size(0)
        correctCC += predictedCC.eq(targetsCC.long()).sum().item()
        totalMLO += targetsMLO.size(0)
        correctMLO += predictedMLO.eq(targetsMLO.long()).sum().item()

        phase_predCC = np.append(phase_predCC, outputsCC[0,1].cpu().numpy())        
        phase_labelCC = np.append(phase_labelCC, targetsCC.data.cpu().numpy())
        phase_predMLO = np.append(phase_predMLO, outputsMLO[0,1].cpu().numpy())
        phase_labelMLO = np.append(phase_labelMLO, targetsMLO.data.cpu().numpy())


    epoch_aucCC = roc_auc_score(phase_labelCC, phase_predCC)
    epoch_aucMLO = roc_auc_score(phase_labelMLO, phase_predMLO)
    epoch_aucAVG = 0.5*(epoch_aucCC + epoch_aucMLO)
    print ('testLossCC: %.3f, testAccuCC: %.3f%% (%d/%d)' % (test_lossCC/(batch_idx+1), 100.*correctCC/totalCC, correctCC, totalCC))
    print ('testLossMLO: %.3f, testAccuMLO: %.3f%% (%d/%d)' % (test_lossMLO/(batch_idx+1), 100.*correctMLO/totalMLO, correctMLO, totalMLO)) 
    print ('testepoch_aucCC: %.3f,testepoch_aucMLO: %.3f,testepoch_aucAVG: %.3f' % (epoch_aucCC,epoch_aucMLO,epoch_aucAVG))
    #print ('testLosscorr: %.3f,testLosstotal: %.3f' % (test_losscorr/(batch_idx+1), test_losstotal/(batch_idx+1)))
    print ('testLosstotal: %.3f' % (test_losstotal/(batch_idx+1)))

    # Save checkpoint.
    accCC = 100.*correctCC/totalCC
    accMLO = 100.*correctMLO/totalMLO
    accAVG = 0.5*(accCC+ accMLO)
    
    if((accAVG > best_accAVG) or (accAVG == best_accAVG and epoch_aucAVG> best_aucAVG)):#if (accAVG > best_accAVG):
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'accCC': accCC,
            'accMLO': accMLO,
            'accAVG': accAVG,
            'epoch_aucCC': epoch_aucCC,
            'epoch_aucMLO': epoch_aucMLO,
            'epoch_aucavg': epoch_aucAVG, 
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, "best.pth")
        best_accCC = accCC
        best_accMLO = accMLO
        best_accAVG = accAVG
        
        best_aucCC = epoch_aucCC
        best_aucMLO = epoch_aucMLO
        best_aucAVG = epoch_aucAVG
        
'''Training'''
for epoch in range(start_epoch, start_epoch+500):
    if (epoch %30 ==0 and epoch!=0):
      adjust_learning_rate(optimizer, decay_rate=0.9)
    train(epoch)
    test(epoch)