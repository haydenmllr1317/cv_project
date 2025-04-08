# imports:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plot
import cv2
import os
from pathlib import Path
import clip
import customDataset
import cus
import numpy as np
import torch.optim as optim
import torchvision.transforms as tv_t
import random
from PIL import Image
import json
import math
from safetensors.torch import load_model, save_model
from models.CLIP_Segmenter import extract_CLIP_features

############################### HYPERPARAMETERS ###############################
epochs = 20
batch_size = 32
loss_fn=torch.nn.CrossEntropyLoss()
sm = nn.Softmax(dim=1)
eps=1e-5
###############################################################################

# FIRST, WE HAVE OUR SLIGHTLY MODIFIED CLIP ARCHITECTURE TO DEAL WITH THE
# PROMPT POINT:

# this is the exact same structure as in CLIP_Segmenter.py, except we double
# the input dimensions (as latent image and point map are concatendated before
# being fed into this model) and our output dimensions is now four to handle
# the fact that we now have four classes:
# [backgroundClicked, catClicked, dogClicked, notClicked]
class SegmentationDecoder(nn.Module):
	def __init__(self,outDim=4):
		super().__init__()

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(512),
			nn.GELU(),

			nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(128),
			nn.GELU(),

			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(64),
			nn.GELU(),

			nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(32),
			nn.GELU(),

			nn.Conv2d(32, outDim, kernel_size=1)
		)

	def forward(self, x,logits=False):
		x=self.decoder(x)
		if(logits):
			return x
		else:
			return F.softmax(x,dim=1)

# this class is also identical to that in CLIP_Segmenter.py except for the
# forward method, in which we now additionally consume a point map (detailed in 
# prompt_based_customDataset) and pass both that and the input image
# through the extract_CLIP_features method (pretrained encoder) before
# concatenating them and passing the result through the CLIP decoder
class ClIP_Segmentation_Model(nn.Module):
  def __init__(self,device):
    super().__init__()
    self.encoderModel, _ = clip.load("ViT-B/16", device=device)
    self.decoderModel = SegmentationDecoder()

  #Helper function to enable grads only for decoder model
  def trainDecoderOnly(self):
    for param in self.encoderModel.parameters():
      param.requires_grad = False
    for param in self.decoderModel.parameters():
      param.requires_grad = True

  #Helper function to enable grads for both models
  def trainBoth(self):
    for param in self.encoderModel.parameters():
      param.requires_grad = True
    for param in self.decoderModel.parameters():
      param.requires_grad = True

  #Helper function to disable grads for both models
  def trainNone(self):
    for param in self.encoderModel.parameters():
      param.requires_grad = False
    for param in self.decoderModel.parameters():
      param.requires_grad = False

  def forward(self, x,p,logits=False):
    x = extract_CLIP_features(x,self.encoderModel)
    x_point = extract_CLIP_features(p,self.encoderModel)
    z = torch.cat((x, x_point), dim=1)
    #Decode features into segmentation mask, using decoder model
    final=self.decoderModel(z,logits=logits)
    return final

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(str(device))

# we load our model onto our computer and
# we set the pretrained encoder to eval mode and train only the decoder portion
clipModel=ClIP_Segmentation_Model(device).to(device)
clipModel.trainDecoderOnly()
clipModel.encoderModel.eval()

# this is a helper method for when we are printing out example photos
# it takes a 1 channel tensor (either the mask or model output) and conv
def output_to_photo(tens):
  color_it = np.array([[255,255,255],[255,255,255],[255,255,255], [0,0,0]], dtype=np.uint8)
  tens = torch.argmax(tens, dim=0).numpy().astype(int)
  return color_it[tens]

# paths to our input training and validation data
train_path = ''
val_path = ''
training_label_path = ''
val_label_path = ''

optimizer = optim.Adam(clipModel.decoderModel.parameters(), lr=0.001)

# grabs CLIPS required input resolution
input_resolution = int(clipModel.encoderModel.visual.input_resolution)

clip_image_norm=tv_t.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

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
train_dataset = customDataset.imageLoaderDataset(dataPairs_Train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

#Dev set dataset/loader
val_dataset = customDataset.imageLoaderDataset(dataPairs_Val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_iter = iter(val_loader)

# lists for tracking loss, IoU and Dice scores
training_loss_list = []
val_loss_list = []
epoch_list = []
train_dice_list = []
train_iou_list = []
val_iou_list = []
val_dice_list = []

for epoch in range(epochs):

    print(str(epoch) + ' started')

    # here we initiate lists for tracking loss and accuracy scores
    running_loss = 0
    count = 0
    count_val = 0
    running_val_loss = 0
    net_sum = 0 # training pixel area sum
    net_intersection = 0 #training pixel are intersection
    net_union = 0 # training pixel area union
    # now the same for validation dataset:
    val_net_sum = 0 
    val_net_intersection = 0
    val_net_union = 0

    for batch_idx, (inputImage,imageClean,targetMask, pointMap) in enumerate(train_loader):

        clipModel.decoderModel.train()
        count += 1

        #Move data to device
        inputImage=inputImage.to(device)
        targetMask=targetMask.to(device)
        pointMap = pointMap.to(device)

        #Bring back into 0-1 range, and apply normalization as required by clip
        with torch.no_grad():
            inputImage=(inputImage+1.0)*0.5
            inputImage=clip_image_norm(inputImage)

        #forward
        outputs = clipModel(inputImage,pointMap,logits=True)
        target_indices = torch.argmax(targetMask, dim=1)
        loss = loss_fn(outputs, target_indices)

        # every other epoch we calculate and track training accuracy metrics
        if (epoch+1)%2 == 0:
            running_loss += loss.cpu().detach().item()
            img = outputs.cpu().detach()
            label = targetMask.cpu().detach()
            img = sm(img)
            # now, we get our output's classes for each pixel:
            choices = torch.argmax(img, dim=1, keepdim=True)
            # we then scatter these classes out to all four channels with 1
            # when that pixel belongs to the class that channel represents
            # and 0 otherwise
            img = torch.zeros_like(img).scatter_(1, choices, 1)
            intersection = torch.sum(img*label,dim=[0,2,3])
            sum = torch.sum(img,dim=[0,2,3]) + torch.sum(label,dim=[0,2,3])
            union = sum - intersection
            
            if net_sum is None:
                net_sum = sum
            else:
                net_sum += sum

            if net_intersection is None:
                net_intersection = intersection
            else:
                net_intersection += intersection
            if net_union is None:
                net_unionn = union
            else:
                net_union += union

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # for ever other epoch, we calculate and track validation loss and metrics
    if (epoch+1)%2 == 0:
        clipModel.decoderModel.eval()
        for _, (v_img_old,_, v_label, v_point) in enumerate(val_loader):
            # Move data to device
            v_img_old = v_img_old.to(device)
            v_label = v_label.to(device)
            v_point = v_point.to(device)
            #Calculate dev loss and other metrics
            with torch.no_grad():
                v_img = (v_img_old + 1.0) * 0.5
                v_img = clip_image_norm(v_img)
                val_result = clipModel(v_img, v_point, logits=True)
                v_target = torch.argmax(v_label, dim=1)
                v_loss = loss_fn(val_result, v_target)
                count_val += 1
                running_val_loss += v_loss.cpu().detach().item()

                # at the final epoch, we save a bunch of photos
                # and output the corresponding model results for inspection
                # these photos are the first photos from every batch in the final epoch
                if epoch+1 == epochs:
                    # we now have our saving paths:
                    og_path = '' # original input photo path
                    label_path = '' # mask path
                    point_path = '' # point map path
                    path = '' # model output path
                    og_photo = v_img_old[0].cpu()
                    og_photo = og_photo + 1
                    og_photo = og_photo*127.5
                    og_photo = og_photo.numpy().transpose(1,2,0)
                    og_photo = og_photo.astype(np.uint8)
                    og_photo = og_photo[..., ::-1]
                    cv2.imwrite(og_path, og_photo)
                    label = (output_to_photo(v_label[0].cpu()))
                    label = (label).astype(np.uint8)
                    label = label[..., ::-1]
                    cv2.imwrite(label_path, label)
                    v_output = (output_to_photo(val_result[0].cpu()))
                    v_output = (v_output).astype(np.uint8)
                    v_output = v_output[..., ::-1]
                    cv2.imwrite(path, v_output)
                    dot_photo = (v_point[0].cpu())*255
                    dot_photo = dot_photo.numpy().transpose(1,2,0).astype(np.uint8)
                    dot_photo = dot_photo[..., ::-1]
                    cv2.imwrite(point_path, dot_photo)

                # now take our result and log our intersection, sum, and 
                # union results
                val_result = val_result.cpu().detach()
                val_result = sm(val_result)
                val_choices = torch.argmax(val_result, dim=1, keepdim=True)
                val_result = torch.zeros_like(val_result).scatter_(1, val_choices, 1)
                v_label = v_label.cpu().detach()
                intersection = torch.sum(val_result*v_label,dim=[0,2,3])
                sum = torch.sum(val_result,dim=[0,2,3]) + torch.sum(v_label,dim=[0,2,3])
                union = sum - intersection
                
                if val_net_sum is None:
                    val_net_sum = sum
                else:
                    val_net_sum += sum

                if val_net_intersection is None:
                    val_net_intersection = intersection
                else:
                    val_net_intersection += intersection
                if val_net_union is None:
                    val_net_union = union
                else:
                    val_net_union += union

        # we now calculate our IoU and Dice scores
        intersectionOverUnion_val = (val_net_intersection + eps) / (val_net_union + eps)
        intersectionOverUnion_val = torch.mean(intersectionOverUnion_val).item()

        dice_val = (2*val_net_intersection + eps) / (val_net_sum + eps)
        dice_val = torch.mean(dice_val).item()

        intersectionOverUnion = (net_intersection + eps) / (net_union + eps)
        intersectionOverUnion = torch.mean(intersectionOverUnion).item()

        dice = (2*net_intersection + eps) / (net_sum + eps)
        dice = torch.mean(dice).item()

        # here we add our loss and accuracy results to our tracking lists
        # and print the results for this given epoch
        train_iou_list.append(intersectionOverUnion)
        train_dice_list.append(dice)
        val_iou_list.append(intersectionOverUnion_val)
        val_dice_list.append(dice_val)
        training_loss_list.append(running_loss/count)
        val_loss_list.append(running_val_loss/count_val)
        epoch_list.append(epoch)
        print('after epoch ' + str(epoch) + ' training loss is: ' + str(running_loss/count))
        print('after epoch ' + str(epoch) + ' val loss is: ' + str(running_val_loss/count_val))
        print('after epoch ' + str(epoch) + ' val iou is: ' + str(intersectionOverUnion_val))
        print('after epoch ' + str(epoch) + ' val dice is: ' + str(dice_val))
        print('after epoch ' + str(epoch) + ' train iou is: ' + str(intersectionOverUnion))
        print('after epoch ' + str(epoch) + ' train dice is: ' + str(dice))


# here we save plots of our loss and classification success rate over epochs
output_images_path = ''

plot.figure()
plot.plot(epoch_list,val_loss_list, marker='o', linestyle='--', color='b', label='Val Loss')
plot.plot(epoch_list, training_loss_list, marker='o', linestyle='--', color='r', label='Train Loss')
plot.xlabel('No. of Epochs')
plot.ylabel('Current Loss')
plot.title('Loss for Pixel-Wise Prompt CLIP')
plot.legend()
plot.grid(True)
plot.savefig(output_images_path + 'prompt_clip_loss.png')

plot.figure()
plot.plot(epoch_list, val_iou_list, marker='o', linestyle='--', color='b', label='Val IoU')
plot.plot(epoch_list, train_iou_list, marker='o', linestyle='--', color='r', label='Train IoU')
plot.xlabel('No. of Epochs')
plot.ylabel('Current IoU')
plot.title(' IoU for Pixel-Wise Prompt CLIP')
plot.legend()
plot.grid(True)
plot.savefig(output_images_path + 'prompt_clip_iou.png')

# finally, we save our model
prompted_clip_path = ''
torch.save(clipModel.state_dict(), prompted_clip_path)
