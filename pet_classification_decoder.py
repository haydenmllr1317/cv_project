# this file is for training and creation of the pet classification decoder for
# the utoencoder implementation (task 2b)

#imports:
import os
import torch
import cv2
import math
import random
import customDataset
from customDataset import imageLoaderDataset
import evalUtil
from evalUtil import get_IoU
from evalUtil import get_dice_coef
from evalUtil import get_pixel_acc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plot
import torch.optim as opt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

############################### HYPERPARAMETERS ###############################
epochs = 10
batch_size = 32
criterion = nn.BCEWithLogitsLoss()
sig = nn.Sigmoid()
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input data paths
train_path = '' # path to raw training data
val_path = '' #path to raw validation data
training_label_path = '' # path to training masks
val_label_path = '' #path to validation masks

# this is the architecture for the self-supervised model that we trained in
# self_supervised_autoencoder.py
# NOTE: We are not traning this here, we are loading a daved encoder
# onto this architecture to use as a fixed, pretrained, encoder
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.MaxPool2d(2, stride=2)

        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=2,stride=2),
            # nn.ReLU() # for first task, just recreating photo so want ReLU, but maybe want Sigmoid for classification later

        )
    def forward(self, _):
        _ = self.encoder(_)
        # print(str(_.size()))
        _ = self.decoder(_)
        return _


# here we load our saved encoder onto this architecture and switch it into
# eval mode
cae = CAE()
encoder_path = ''
encoder = cae.encoder.to(device)
encoder.load_state_dict(torch.load(encoder_path))
encoder.eval()

# this is our binary classification architecture, named "Decider" for the fact
# that is being trained to decide whether an image contains a cat or a dog
class Decider(nn.Module):
    def __init__(self):
        super(Decider, self).__init__()
        self.cov = nn.Sequential(

            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)

        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,1),
        )

    def forward(self,_):
        _ = self.cov(_)
        _ = self.mlp(_)
        return _
    
# this takes the target (mask) and returns a list (of batch size length)
# of 1.0's or 0.0's depending on if a mask contains a cat or dog
def binary_animal(inputMask):
  # channel 1 is cat, channel 2 is dog, summing over to see which is present
  photo_animal=torch.sum(inputMask[:,1:2,:,:],dim=(1,2,3))>torch.sum(inputMask[:,2:3,:,:],dim=(1,2,3))
  return photo_animal.to(torch.float) # true, aka 1.0 is catmask

# checks in a batch of 32 output images, how many of them correctly classify
# which animal is present (what fraction)
def correctness(output, target):
  correct = 0
  for i in range(output.size()[0]):
    x = output[i].cpu().detach().numpy()
    y = target[i].cpu().detach().numpy()
    if np.round(x) == int(y):
      correct += 1
  return correct/(output.size()[0])

# here we create our dataset objects
# first, we just have our input images, we add the masks in the for loops to follow
dataPairs_Train=customDataset.get_files_in_folder(train_path)
dataPairs_Train.sort()
random.seed(0)
random.shuffle(dataPairs_Train)
dataPairs_Val = customDataset.get_files_in_folder(val_path)
dataPairs_Val.sort()
random.seed(0)
random.shuffle(dataPairs_Val)

#create input-output pairs
for i in range(len(dataPairs_Train)):
  #Labels seem to always be pngs
  labelImageName=Path(dataPairs_Train[i]).stem+".png"
  dataPairs_Train[i]=(dataPairs_Train[i],os.path.join(training_label_path, labelImageName))
for i in range(len(dataPairs_Val)):
  #Labels seem to always be pngs
  labelImageName=Path(dataPairs_Val[i]).stem+".png"
  dataPairs_Val[i]=(dataPairs_Val[i],os.path.join(val_label_path, labelImageName))

#Train set dataset/loader
train_dataset = imageLoaderDataset(dataPairs_Train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

#Dev set dataset/loader
val_dataset = imageLoaderDataset(dataPairs_Val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_iter = iter(val_loader)


# here we initiate our "decider" (pet classification) model
decider = Decider()
decider = decider.to(device)

optimizer = opt.Adam(decider.parameters(), lr=0.0001)

epoch_list = [] # list of passing epochs, for plots
val_loss_list = [] # list of validation loss, for plots
training_loss_list = [] # list of training loss, for plots
val_perc_list = [] # list of validation succesful classification percentages
training_perc_list = [] # list of training succesful classification percentages

# this is the training and validation loop for our model
for epoch in range(epochs):
    print(str(epoch) + ' started')
    decider.train()
    net_training_loss = 0
    net_training_perc = 0
    train_n = 0
    for batch_num, (img,_,label) in enumerate(train_loader):
        train_n += 1
        img,label = img.to(device), label.to(device)
        optimizer.zero_grad() # zeroes out grad for training
        latent = encoder(img) # pretrained encoder
        result = decider(latent)
        loss = criterion(result.squeeze(dim=1), binary_animal(label))
        net_training_loss += loss.cpu().detach().item()
        net_training_perc += correctness(sig(result.squeeze(dim=1)), binary_animal(label))
        loss.backward()
        optimizer.step()

    # every other epoch we run through the validation data
    if (epoch+1) % 2 == 0:
        decider.eval()
        n = 0
        net_loss = 0
        net_val_perc = 0
        for v_img,_,v_label in val_loader:
            n += 1
            v_img, v_label = v_img.to(device), v_label.to(device)
            v_latent = encoder(v_img)
            val_result = decider(v_latent)

            # at the final epoch, we save a bunch of photos
            # and output the corresponding model results for inspection
            # these photos are the first photos from every batch in the final epoch
            if epoch+1 == epochs:
                og_path = ''
                v_photo = v_img[0].cpu().detach().numpy().transpose(1,2,0)
                v_photo = (v_photo*255).astype(np.uint8)
                v_photo = v_photo[..., ::-1] # as cv2 uses the opposite color order
                cv2.imwrite(og_path, v_photo)
                # now we print what pet was predicted (in binary)
                print('Pet ' + str(n) + ': ' + str(sig(val_result.squeeze(dim=1)[0]).cpu().detach().numpy()))
                # and what pet was the correct answer
                print('Correct Pet ' + str(n) + ': ' + str(binary_animal(v_label)[0].cpu().detach().numpy()))

            # now we calculate our validation loss and accuracy rate and track them
            val_loss = criterion(val_result.squeeze(dim=1), binary_animal(v_label))
            net_loss += val_loss.cpu().detach().item()
            net_val_perc += correctness(sig(val_result.squeeze(dim=1)), binary_animal(v_label))

        # here we add our loss and accuracy results to our tracking lists
        # and print the results for this given epoch
        epoch_list.append(epoch)
        training_loss_list.append(net_loss/n)
        training_loss_list.append(net_training_loss/train_n)
        training_perc_list.append(net_training_perc/train_n)
        val_perc_list.append(net_val_perc/n)
        print('after epoch ' + str(epoch) + ' training loss is ' + str(net_training_loss/train_n))
        print('after epoch ' + str(epoch) + ' train fraction is ' + str(net_training_perc/train_n))
        print('after epoch ' + str(epoch) + ' val loss is ' + str(net_loss/n))
        print('after epoch ' + str(epoch) + ' val fraction is ' + str(net_val_perc/n))

# here we save plots of our loss and classification success rate over epochs
output_images_path = ''
plot.figure()
plot.plot(epoch_list,val_loss_list, marker='o', linestyle='--', color='b', label='Val Loss')
plot.plot(epoch_list, training_loss_list, marker='o', linestyle='--', color='r', label='Train Loss')
plot.xlabel('No. of Epochs')
plot.ylabel('Current Loss')
plot.title('Loss for Pet Classifier')
plot.legend()
plot.grid(True)
plot.savefig(output_images_path + 'decider_loss.png')

plot.figure()
plot.plot(epoch_list, val_perc_list, marker='o', linestyle='--', color='b', label='Val Fraction')
plot.plot(epoch_list, training_perc_list, marker='o', linestyle='--', color='r', label='Train Fraction')
plot.xlabel('No. of Epochs')
plot.ylabel('Classification Percentage')
plot.title('Classifier Accuracy Fraction')
plot.legend()
plot.grid(True)
plot.savefig(output_images_path + 'decider_perc.png')

# finally, we save our model
decider_save_path = ''
torch.save(decider.state_dict(), decider_save_path)