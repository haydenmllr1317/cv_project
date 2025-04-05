import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as tv_t
import random
import math

################################################################################
# THIS IS A VERSION OF OUR DATASET CREATION CODE THAT IS MODIFIED TO SUPPORT   #
# POINT-PROMPT BASED SEGMENTATION. THIS FILE IS ALMOST IDENTICAL TO            #
# customDataset.py EXCEPT AT A FEW LOCATIONS, WHICH I WILL HIGHLIGHT WITH      #
# NOTABLE COMMENTS using the NOTE: format                                      #
################################################################################


#Loads an image from a given path. (range: 0 to 255)
#If image has alpha channel, replaces it with white
def loadIm(targetPath):	#in range 0.0-255.0
	with torch.no_grad():
		with Image.open(targetPath) as im:
			#Handle transparency
			if im.mode=="RGBA":
				im.load()
				background = Image.new("RGB", im.size, (255, 255, 255))
				background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
				im2=background
			else:
				if(im.mode!="RGB"):
					im2=im.convert("RGB")
				else:
					im2=im
			imageTensor=torch.moveaxis(torch.Tensor(np.array(im2,copy=True)),2,0)
			return imageTensor[None]

#Scales the shortest side of the image to target size, then random crops the other dim to match
#Performs matching scale/crop to mask (if present)
def resizeImage(inputImage,inputMask=None,targetWidth=128,targetHeight=128):
	with torch.no_grad():
		if(inputImage.size(2)>inputImage.size(3)):#height bigger than width
			#Resize if needed
			if(inputImage.size(3)!=targetWidth):
				newWidth=targetWidth
				newHeight=math.ceil(inputImage.size(2)*(targetWidth/inputImage.size(3)))


				inputImage=F.interpolate(inputImage,size=(newHeight,newWidth),mode="bilinear",antialias="bilinear")
				if(not(inputMask is None)):
					inputMask=F.interpolate(inputMask,size=(newHeight,newWidth),mode="bilinear",antialias="bilinear")

			#Get random crop offset
			randomOffset=random.randint(0,inputImage.size(2)-targetHeight)

			#Do the crop
			inputImage=inputImage[:,:,randomOffset:randomOffset+targetHeight,:]
			if(not(inputMask is None)):
				inputMask=inputMask[:,:,randomOffset:randomOffset+targetHeight,:]

		else:
			#Resize if needed
			if(inputImage.size(2)!=targetHeight):
				newWidth=math.ceil(inputImage.size(3)*(targetHeight/inputImage.size(2)))
				newHeight=targetHeight

				inputImage=F.interpolate(inputImage,size=(newHeight,newWidth),mode="bilinear",antialias="bilinear")
				if(not(inputMask is None)):
					inputMask=F.interpolate(inputMask,size=(newHeight,newWidth),mode="bilinear",antialias="bilinear")

			#Get random crop offset
			randomOffset=random.randint(0,inputImage.size(3)-targetWidth)

			#Do the crop
			inputImage=inputImage[:,:,:,randomOffset:randomOffset+targetWidth]
			if(not(inputMask is None)):
				inputMask=inputMask[:,:,:,randomOffset:randomOffset+targetWidth]

		return inputImage,inputMask

#Pass to the dataloader to stop it from trying to merge the tensors automatically
def custom_collate(original_batch):
  return original_batch


# NOTE: Now, this method takes in two additional inputs, x and y, which
#       correspond to the selected pixel coordinates
# Now the method:
# Takes in mask images where:
#	Black is background
#	Red is cat
#	Green is dog
#	White is border
# And turns then into a mask where:
#	Channel 0 is background was clicked (at this pixel)
#	Channel 1 is cat was clicked (at this pixel)
#	Channel 2 is dog was clicked (at this pixel)
#	Channel 3 is nothing was clicked (at this pixel)
#	(border pixels are delt with in the same way as before, priort to
#    incorporating the prompt)
def HandleMaskConversion(inputMask,x,y):
  #Get white pixels (border)
  borderMask=(inputMask[:,2:3,:,:]>0.5).to(torch.float32)

  #Get dog/cat pixels, (and exlude the border)
  catMask=(inputMask[:,0:1,:,:]>0.5).to(torch.float32)
  dogMask=(inputMask[:,1:2,:,:]>0.5).to(torch.float32)
  catMask=catMask*(1.0-borderMask)
  dogMask=dogMask*(1.0-borderMask)

  #Check whether most of the pixels are cat or dog
  isMoreCat=torch.sum(catMask)>torch.sum(dogMask)

  #Turn the border pixels into cat/dog, depending on what is most present in the image
  if(isMoreCat):
    catMask=torch.max(catMask,borderMask)
  else:
    dogMask=torch.max(dogMask,borderMask)

  #Background is anything not taken by cat/dog mask
  backgroundMask=1-torch.max(catMask,dogMask)

  catClicked = torch.zeros_like(catMask)
  dogClicked = torch.zeros_like(catMask)
  bgClicked = torch.zeros_like(catMask)

  # NOTE: as a point can only correspond to one class, we now find out which
  # class the clicked point belonged to, and set that classes "clicked" value
  # to 1.0
  if catMask[:,:,x,y].item() == 1.0:
    # print('are we here')
    catClicked = catMask
    # print('we are')
  elif dogMask[:,:,x,y].item() == 1.0:
    # print('are we heddre')
    dogClicked = dogMask
    # print('w are')
  elif backgroundMask[:,:,x,y].item() == 1.0:
    # print('are we adhere')
    bgClicked = backgroundMask
    # print(' aw are')
  else:
    raise ValueError('nothing was clicked')

  # everything that isn't part of a clicked object is "not clicked"
  nClicked = 1-(catClicked + dogClicked + bgClicked)
  
  # returns our new 4 channel tensor
  return torch.cat([bgClicked, catClicked, dogClicked, nClicked], dim=1)

def AugmentImage(inputImage,inputMask=None,skipAugments=False):
  embed_xFlip=0
  embed_yFlip=0
  embed_rotAmount=0
  embed_hueShift=0


  augment_chance=1.0/4

  #Make a clean copy of the image
  imageClean=inputImage+0

  if(not skipAugments):
    #X-flip:
    if(bool(random.getrandbits(1))):
      embed_xFlip=1
      inputImage=inputImage.flip(3)
      imageClean=imageClean.flip(3)
      if(not(inputMask is None)):
        inputMask=inputMask.flip(3)

    #Y-flip:
    if(random.uniform(0, 1)<augment_chance):
      if(bool(random.getrandbits(1))):
        embed_yFlip=1
        inputImage=inputImage.flip(2)
        imageClean=imageClean.flip(2)
        if(not(inputMask is None)):
          inputMask=inputMask.flip(2)

    #Rotate:
    if(random.uniform(0, 1)<augment_chance):
      embed_rotAmount=random.uniform(-1,1)
      inputImage=tv_t.functional.rotate(inputImage,embed_rotAmount*90)
      imageClean=tv_t.functional.rotate(imageClean,embed_rotAmount*90)
      if(not(inputMask is None)):
        inputMask=tv_t.functional.rotate(inputMask,embed_rotAmount*90)

    #Hue shift:
    if(random.uniform(0, 1)<augment_chance):
      embed_hueShift=random.uniform(-1,1)
      inputImage=(inputImage+1)*0.5
      inputImage=tv_t.functional.adjust_hue(inputImage,embed_hueShift*0.5)
      inputImage=(inputImage*2)-1

  return inputImage,imageClean,inputMask,[embed_xFlip,embed_yFlip,embed_rotAmount,embed_hueShift]


class imageLoaderDataset(torch.utils.data.Dataset):
  def __init__(self, dataPairs,skipAugments=False,targetRes=128):
    #Initialization
    self.dataPairs = dataPairs
    self.skipAugments=skipAugments
    self.targetRes=targetRes


  def __len__(self):
    return len(self.dataPairs)

  def __getitem__(self, index):
    try:
      with torch.no_grad():

        #Load image/mask
        loadedImage=loadIm(self.dataPairs[index][0])
        loadedMask=loadIm(self.dataPairs[index][1])

        #Resize/crop both
        loadedImage,loadedMask=resizeImage(loadedImage,loadedMask,targetWidth=self.targetRes,targetHeight=self.targetRes)

        #Adjust image range to -1,1
        loadedImage=(loadedImage/127.5)-1.0

        #Adjust mask range to 0,1
        loadedMask=torch.round((loadedMask/255.0))

        # NOTE: this is where we generate our randomly selected prompt point
        # form the size of the input images
        random_x = torch.randint(0,self.targetRes, (1,)).item()
        random_y = torch.randint(0,self.targetRes, (1,)).item()

        # now we create a tensor of zeroes to match the size of the input images 
        point_map = torch.zeros(3, self.targetRes, self.targetRes)
        # and set the prompt point to 1.0 in all channels
        point_map[:, random_x, random_y] = 1.0 

        loadedMask=HandleMaskConversion(loadedMask, random_x, random_y)

        #Add random augmentations
        loadedImage,imageClean,loadedMask,_=AugmentImage(loadedImage,loadedMask,skipAugments=self.skipAugments)

        # NOTE: __getitem__, which is employed by Dataloader to grab dataset
        #      elements, now returns an additional element, point_map
        #      which is a 3-channel image-sized tensor of all zeros except
        #      at the prompt point where the value is 1 (for all channels)
        return loadedImage[0],imageClean,loadedMask[0], point_map

    except Exception as ex:
      print(ex)
      return None,None