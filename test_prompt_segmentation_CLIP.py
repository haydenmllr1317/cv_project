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
from customDataset import imageLoaderDataset
import cus
import numpy as np
import torch.optim as optim
import torchvision.transforms as tv_t
import random
from PIL import Image
import json
import math

## THIS FILE IS FOR TESTING THE prompt_segmentation_clip.py FILE

# FIRST, we include the architecture of the model:

#Extracts the image tokens from a clip model, rather than just the class token.
#(bit hacky but couldn't find a cleaner method in the CLIP repo)
#Should match the code in:
#	https://github.com/openai/CLIP/blob/main/clip/model.py
#		clip/model.py -> VisionTransformer -> forward
def extract_CLIP_features(image_batch,model):
	#The model seems to be loaded as f16.
	#Keep note of the input dtype, and temporarely cast it to fp16 for the clip pass.
	dtype_original=image_batch.dtype
	image_batch=image_batch.to(model.dtype)

	vision_model = model.visual
	x = vision_model.conv1(image_batch)#[batch, width, grid, grid]
	batch, width, grid = x.shape[0], x.shape[1], x.shape[2]

	#prepare for transformer
	x = x.reshape(batch, width, -1).permute(0, 2, 1)  # [batch, grid^2, width]
	x = torch.cat([vision_model.class_embedding.expand(batch, 1, -1), x], dim=1)
	x += vision_model.positional_embedding
	x = vision_model.ln_pre(x)

	#run through transformer
	x = x.permute(1, 0, 2)  #[seq_len, batch, width]
	#Cast dtype again, the previous step seems to ruin it
	#(possibly because only some of the clip weights are quantized)
	x=x.to(model.dtype)
	x = vision_model.transformer(x)
	x = x.permute(1, 0, 2)  #[batch, seq_len, width]

	#get clip features (ignore class token)
	patch_embeddings = x[:, 1:, :]  #[batch, grid^2, width]

	#apply final projection
	#TODO: not entirely sure if this is needed, check again later
	patch_embeddings = vision_model.ln_post(patch_embeddings)
	if vision_model.proj is not None:
		patch_embeddings = patch_embeddings @ vision_model.proj

	#un-flatten back into width/height
	output=patch_embeddings.permute(0, 2, 1).view(batch, -1, grid, grid)

	#Cast back into our original dtype
	output=output.to(dtype_original)
	return output


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

# here we load the save CLIP model (for segmentation) we are testing
clipModel=ClIP_Segmentation_Model(device).to(device)
clip_path = ''
clipModel.load_state_dict(torch.load(clip_path))
clipModel.trainNone()

# this method consumes a tensor with four channels
# these are [background clicked, cat clicked, dog clicked, none clicked]
# and it paints the clicked region white and leaves the rest black
# this is just for inspecting photos after testing
def output_to_photo(tens):
  color_it = np.array([[255,255,255],[255,255,255],[255,255,255], [0,0,0]], dtype=np.uint8)
  tens = torch.argmax(tens, dim=0).numpy().astype(int)
  return color_it[tens]


batch_size = 32
test_path = ''
test_mask_path = ''

# here are our testing images
test_pairs=customDataset.get_files_in_folder(test_path)
test_pairs.sort()
random.seed(0)
random.shuffle(test_pairs)

# resolution required by CLIP
input_resolution = int(clipModel.encoderModel.visual.input_resolution)
# input_resolution = 224

clip_image_norm=tv_t.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

# attach masks to images to form dataset pairs
for i in range(len(test_pairs)):
  labelImageName=Path(test_pairs[i]).stem+".png"
  test_pairs[i]=(test_pairs[i], os.path.join(test_mask_path, labelImageName))

test_dataset = imageLoaderDataset(test_pairs, skipAugments=True, targetRes=input_resolution)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=False)

# NOW OUR TESTING LOOP:

net_PA = 0
n = 0

eps=1e-5

net_intersection = None
net_union = None
net_sum = None


with torch.no_grad():
    # loop through the testing dataloader:
    for img_old,_, label, points in test_loader:
        n += 1
        print(n)
        # Move data to device
        img_old = img_old.to(device)
        label = label.to(device)
        points = points.to(device)
        #calculate loss, 
        img = (img_old + 1.0) * 0.5
        img = clip_image_norm(img)
        result = clipModel(img, points, logits=True)

        # now we pring out the first image from each batch of the original input
        # photo (og_photo), the label (mask), the output photo (output), and the
        # dot_photo which is a black photo with a white pixel corresponding
        # to the point used to segment the image
        og_path = ''
        label_path = ''
        point_path = ''
        path = ''
        og_photo = img_old[0].cpu()
        og_photo = og_photo + 1
        og_photo = og_photo*127.5
        og_photo = og_photo.numpy().transpose(1,2,0)
        og_photo = og_photo.astype(np.uint8)
        og_photo = og_photo[..., ::-1]
        cv2.imwrite(og_path, og_photo)
        mask = (output_to_photo(label[0].cpu()))
        mask = (mask).astype(np.uint8)
        mask = mask[..., ::-1]
        cv2.imwrite(label_path, mask)
        output = (output_to_photo(result[0].cpu()))
        output = (output).astype(np.uint8)
        output = output[..., ::-1]
        cv2.imwrite(path, output)
        dot_photo = (points[0].cpu())*255
        dot_photo = dot_photo.numpy().transpose(1,2,0).astype(np.uint8)
        dot_photo = dot_photo[..., ::-1]
        cv2.imwrite(point_path, dot_photo)

        # now we calculate accuracy metrics and track them:
        output_choice=torch.argmax(result, dim=1)
        label_choice=torch.argmax(label, dim=1)
        mean_ac = torch.mean((torch.abs(output_choice-label_choice)<0.5).to(torch.float32))
        net_PA += mean_ac.item()

        result = result.cpu().detach()
        choices = torch.argmax(result, dim=1, keepdim=True)
        result = torch.zeros_like(result).scatter_(1, choices, 1)
        label = label.cpu().detach()
        intersection = torch.sum(result*label,dim=[0,2,3])
        sum = torch.sum(result,dim=[0,2,3]) + torch.sum(label,dim=[0,2,3])
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
            net_union = union
        else:
            net_union += union

    # calculate the commmulative IoU and Dice Scores
    intersectionOverUnion = (net_intersection + eps) / (net_union + eps)
    intersectionOverUnion = torch.mean(intersectionOverUnion).item()

    dice = (2*net_intersection + eps) / (net_sum + eps)
    dice = torch.mean(dice).item()

# print out our final testing scores
print('test iou score is ' + str(intersectionOverUnion))
print('test dice score is ' + str(dice))
print('test PA score is ' + str(net_PA/n))