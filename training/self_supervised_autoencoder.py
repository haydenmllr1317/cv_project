# imports
import os
import torch
import cv2
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plot
import torch.optim as opt
import torchvision
import customDataset
from customDataset import imageLoaderDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from models.AutoEnc import CAE

# THIS FILE IS FOR TRAINING THE SELF-SUPERVISED AUTOENCODER FOR PART B)
# OF THE IMAGE SEGMENTATION TASK
# THE ENCODER HERE IS SAVED, AND USED FOR THE REST OF PART B) IN WHICH WE 
# TRAIN A DECODER TO COMPLETE THE SEGMENTATION TASK

batch_size = 32

# these are to be filled in when using
train_path = ''
val_path = ''
training_label_path = ''
val_label_path = ''
output_images_path = ''

epochs = 50
criterion1 = nn.MSELoss()

# this grabs lists of our training and validation input images
dataPairs_Train=customDataset.get_files_in_folder(train_path)
dataPairs_Train.sort()
random.seed(0)
random.shuffle(dataPairs_Train)
dataPairs_Val = customDataset.get_files_in_folder(val_path)
dataPairs_Val.sort()
random.seed(0)
random.shuffle(dataPairs_Val)

# this creates our input, mask pairs
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


# create model and mount to GPU if possible:
cae = CAE()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cae = cae.to(device)

optimizer = opt.Adam(cae.parameters(), lr=0.0001)

epoch_list = []
loss_list = []
training_loss_list = []

# here is our training loop:
for epoch in range(epochs):
    print(str(epoch) + ' started')
    train_n = 0
    net_training_loss = 0
    cae.train()
    for batch_num, (inputs,_,_) in enumerate(train_loader):
        train_n += 1
        img = inputs.to(device)
        optimizer.zero_grad() # this clears gradient from previous run for a fresh start
        result = cae(img)
        loss = criterion1(result, img)
        net_training_loss += loss.item()
        loss.backward()
        optimizer.step()
    cae.eval()
    if (epoch+1) % 2 == 0:
        net_loss = 0
        try:
          (v_img,_,_) = next(val_iter)
        except StopIteration:
          #Reset iterator if reached the end
          val_iter = iter(val_loader)
          (v_img,_,_) = next(val_iter)
        # for inputs,_ in auto_loaded_val:
        v_img = inputs.to(device)
        val_result = cae(v_img)
        if epoch+1 == epochs:
            # save final round of validation images for inspection
            transform = transforms.ToPILImage()
            for i in range((val_result.shape[0])):
              photo = transform(val_result[i])
              val_img = v_img[i]
              val_img = val_img.cpu().detach().numpy().transpose(1, 2, 0)
              val_img = val_img[..., ::-1]
              val_img = (val_img * 255).astype(np.uint8)
              path = output_images_path + str(i) + '_val.png'
              og_path = output_images_path + str(i) + '_og.png'
              cv2.imwrite(og_path,val_img)
              photo.save(path)
        # record val loss and pring it
        val_loss = criterion1(val_result, v_img)
        print('val loss is ' + str(val_loss.cpu().detach()))
        net_loss += val_loss.item()
        epoch_list.append(epoch)
        loss_list.append(net_loss)
        training_loss_list.append(net_training_loss/train_n)
        print('after epoch ' + str(epoch) + ' val loss is' + str(net_loss))
        print('after epoch ' + str(epoch) + ' training loss is ' + str(net_training_loss/train_n))


# plot loss graph!
plot.plot(epoch_list, training_loss_list, marker='o', linestyle='--', color='r', label='Train Loss')
plot.plot(epoch_list,loss_list, marker='o', linestyle='--', color='b', label='Val Loss')
plot.xlabel('No. of Epochs')
plot.ylabel('Current Val Loss')
plot.title('Val Loss Every 10 Epochs')
plot.legend()
plot.grid(True)
plot.savefig(output_images_path + 'cae_loss.png')

# save ONLY the encoder for use in the rest of this task
encoder_save_path = '' # fill this in for use
torch.save(cae.encoder.state_dict(), encoder_save_path + "new_encoder.pth")
