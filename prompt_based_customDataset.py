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
from customDataset import loadIm,resizeImage,AugmentImage

################################################################################
# THIS IS A VERSION OF OUR DATASET CREATION CODE THAT IS MODIFIED TO SUPPORT   #
# POINT-PROMPT BASED SEGMENTATION. THIS FILE IS ALMOST IDENTICAL TO            #
# customDataset.py EXCEPT AT A FEW LOCATIONS, WHICH I WILL HIGHLIGHT WITH      #
# NOTABLE COMMENTS using the NOTE: format                                      #
################################################################################


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
