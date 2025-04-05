# this file is for training and creation of the segmentation decoder for the 
# autoencoder implementation (task 2b)
# this is set to our current implementation of binary classifcation
#   (pet vs background)
# but can be easily chaged to a trinary classification task (3 output channels)

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
epochs = 50
batch_size = 32
criterion = nn.BCEWithLogitsLoss()
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

# here is the model structure for the binary pixel-wise segmentation
# decoder that we are actually training here
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=2)

        )
    def forward(self, _):
        _ = self.decoder(_)
        return _

# this method takes our mask, and alters it for the binary classification
# task by changing it create a one-channel mask that has binary elements
# that indicate a pixel belongs to a pet or it doesnt
def to_animal(label):
 # input: tensor of the form: (batch_size, [background, cat, dog], heigh, width)
 # want to convert to tensor of the form (batch_size, heigh, width) where the value
 # is binary for if there is an object there
  cat_mask = label[:,1:2,:,:]
  dog_mask = label[:,2:3,:,:]
  return torch.max(cat_mask, dog_mask)

# this method consumes a model output (one channel mask)
# and converts it to a mask in which each pixel is 1 or 0
# depending if the model believes the pixel belongs to a pet
def pixel_prediction(output):
  sig = nn.Sigmoid() #use sigmoid for binary classification (softmax for trinary)
  output = sig(output)
  output = output.cpu().detach() # don't want to be on the GPU or track weights
  output = torch.round(output)
  return output


# this function is for saving our outputs, it converts the outputted mask
# (which has one channel with pixel values of 1 or 0) to an RGB image
# that colors the "pet" pixels green and the background pixels black
def output_to_img(output):
  sig = nn.Sigmoid()
  color_it = np.array([[0,0,0],[0,255,0]], dtype=np.uint8)
  output = sig(output)
  output = output.cpu().detach().numpy()
  output = np.round(output)
  output = output.astype(int)
  return color_it[output] # this sends the 1's to green and 0's to black, as desired

# this function takes our mask (called label here) and switches it from
# one channel binary values to three channel "color" values, sending
# pet pixels to green and background pixels to black
def label_to_img(label):
  color_it = np.array([[0,0,0], [0,255,0]])
  label = label.astype(int)
  return color_it[label]

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


# training cell for decoder

decoder = Decoder()
decoder = decoder.to(device)

optimizer = opt.Adam(decoder.parameters(), lr=0.0001)

epoch_list = [] # list of passing epochs, for plots
val_loss_list = [] # list of validation loss, for plots
training_loss_list = [] # list of training loss, for plots
val_iou_list = [] # list of validation IoU, for plots
train_iou_list = [] # list of traiing IoU, for plots


# this is the training and validation loop for our model
for epoch in range(epochs):
    print(str(epoch) + ' started')
    decoder.train()
    net_training_loss = 0 # total training loss for this given epoch
    train_n = 0 # number of batches/epoch
    net_train_iou = 0  # total training iou for this epohc
    for batch_num, (img,_,label) in enumerate(train_loader):
    # for img,label in dec_loaded_train:
        train_n += 1
        img,label = img.to(device), label.to(device)
        optimizer.zero_grad() # removes gradients for new training round
        latent = encoder(img) # this is the previously saved encoder
        result = decoder(latent) 
        loss = criterion(result, to_animal(label))
        net_training_loss += loss.item()
        net_train_iou += get_IoU(pixel_prediction(result), to_animal(label).cpu()).detach().item()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 2 == 0: # try validation dataset every other epoch
        decoder.eval()
        net_loss = 0
        net_val_iou = 0

        try:
          (v_img,_,v_label) = next(val_iter)
        except StopIteration:
          #Reset iterator if reached the end
          val_iter = iter(val_loader)
          (v_img,_,v_label) = next(val_iter)

        v_img, v_label = v_img.to(device), v_label.to(device)
        v_latent = encoder(v_img)
        val_result = decoder(v_latent)

        # now, for the final epoch, we save a bunch of examples for inspection
        if epoch+1 == epochs:
            for i in range((val_result.shape[0])):
              # photo here is our model's predicted segmentation
              photo = output_to_img(val_result[i].squeeze(dim=0))
              output_path = ''
              og_path = ''
              og_path_label = ''
              photo = photo[..., ::-1] # because cv2 does BGR
              v_photo = v_img[i].cpu().detach().numpy().transpose(1,2,0)
              v_photo = (v_photo*255).astype(np.uint8)
              v_photo = v_photo[..., ::-1]
              v_mask = label_to_img(to_animal(v_label)[i].squeeze(dim=0).cpu().detach().numpy())
              v_mask = v_mask[..., ::-1]
              cv2.imwrite(og_path, v_photo)
              cv2.imwrite(output_path,photo)
              cv2.imwrite(og_path_label, v_mask)

        # here we just calculate the loss and iou and print out the result
        # or add them to our tracking lists for our given epoch
        val_loss = criterion(val_result, to_animal(v_label))
        net_loss += val_loss.item()
        net_val_iou += get_IoU(pixel_prediction(val_result), to_animal(v_label).cpu()).detach().item()
        epoch_list.append(epoch)
        val_loss_list.append(net_loss)
        training_loss_list.append(net_training_loss/train_n)
        val_iou_list.append(net_val_iou)
        train_iou_list.append(net_train_iou/train_n)
        print('after epoch ' + str(epoch) + ' training loss is ' + str(net_training_loss/train_n))
        print('after epoch ' + str(epoch) + ' train iou score is ' + str(net_train_iou/train_n))
        print('after epoch ' + str(epoch) + ' val loss is ' + str(net_loss))
        print('after epoch ' + str(epoch) + ' val iou score is ' + str(net_val_iou))
    torch.cuda.empty_cache()


# here is where we plot our training+validation loss and IoU curves
output_images_path = ''
plot.figure()
plot.plot(epoch_list,val_loss_list, marker='o', linestyle='--', color='b', label='Val Loss')
plot.plot(epoch_list, training_loss_list, marker='o', linestyle='--', color='r', label='Train Loss')
plot.xlabel('No. of Epochs')
plot.ylabel('Loss')
plot.title('Loss for Segmentation Decoder')
plot.legend()
plot.grid(True)
plot.savefig(output_images_path + 'binary_class_decoder_loss.png')

plot.figure()
plot.plot(epoch_list, val_iou_list, marker='o', linestyle='--', color='b', label='Val IoU')
plot.plot(epoch_list, train_iou_list, marker='o', linestyle='--', color='r', label='Train IoU')
plot.xlabel('No. of Epochs')
plot.ylabel('IoU')
plot.title('IoU for Segmentation Decoder')
plot.legend()
plot.grid(True)
plot.savefig(output_images_path + 'binary_class_decoder_iou.png')

# save the model
decoder_path = ''
torch.save(decoder.state_dict(), decoder_path)