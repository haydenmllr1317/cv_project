
import math
import random

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

#TODO:
#    Find reference to paper i based the augments on



def loadIm(targetPath):
    """
    Loads an image from a given path.  (range: 0 to 255)
    If image has alpha channel, replaces it with white

    Args:
        targetPath (string): path to an image

    Returns:
        (Tensor): Tensor of size (1,3,height,width) representing an image in the range 0-255
    """

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

def resizeImage(inputImage,inputMask=None,targetWidth=128,targetHeight=128):
    """
    Scales the shortest side of the image to target size, then random crops the other dim to match
    Performs matching scale/crop to mask (if present)

    Args:
        inputImage (Tensor): Tensor of size (1,3,height,width) representing an image.
        inputMask (Tensor): Optional Tensor same size as image representing a segmentation mask.
        targetWidth (int): Target width to rescale to
        targetHeight (int): Target height to rescale to

    Returns:
        (Tensor): Scaled image
        (Tensor): Scaled mask (if given)
    """

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




def HandleMaskConversion(inputMask):
    """
    Converts the segmentation masks that come with the dataset,
        into the format needed by the training scripts

    Specificaly, takes in mask images where:
        Black is background
        Red is cat
        Green is dog
        White is border
    And turns then into a one-hot encoded segmentation mask where:
        Channel 0 is background
        Channel 1 is cat
        Channel 2 is dog
        (border pixels are converted into either cat or dog, depending on what was most present in the image)

    Args:
        inputMask (Tensor): Tensor of size (1,3,height,width) holding a segmentation mask from the original dataset in the range 0-1.

    Returns:
        (Tensor): Converted Mask
    """

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

    return torch.cat([backgroundMask,catMask,dogMask],dim=1)


class imageLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, dataPairs,skipAugments=False,targetRes=128):
        """
        Args:
            dataPairs (list): List of string tuples holding the string path for each image and its mask respectively.
            skipAugments (bool): Whether or not to skip applying augmentations
            targetRes (int): The base resolution to rescale images to.

        """
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

                #Convert mask to the right format:
                #Channel 0 = Background
                #Channel 1 = Cat
                #Channel 2 = Dog
                loadedMask=HandleMaskConversion(loadedMask)

                #Add random augmentations
                loadedImage,imageClean,loadedMask,_=AugmentImage(loadedImage,loadedMask,skipAugments=self.skipAugments)

            #return self.dataPairs[index],loadedImage,loadedMask
            return loadedImage[0],imageClean[0],loadedMask[0]

        except Exception as ex:
            print(ex)
            #print("hi")
            #input()
            #return self.dataPairs[index],None,None
            return None,None



def AugmentImage(inputImage,inputMask=None,skipAugments=False):
    """
    Augments a given image and optionaly applies the appropriate matching edits to its mask.

    Args:
        inputImage (Tensor): an image tensor, size=(1,3,height,width) in the range -1 to 1
        inputMask (Tensor): an optional segmentation mask, same size as the image to apply matching transforms to
        skipAugments (bool): whether or not to skip augmentations

    Returns:
        (Tensor): Augmented Image
        (Tensor): Partialy Augmented image as described in report
        (Tensor): Segmentation mask with matching augmentations applied
        (list): A list of floats representing an embeding of the augmentations applied
    """
    embed_xFlip=0
    embed_yFlip=0
    embed_rotAmount=0
    embed_hueShift=0
    embed_rangeCompress=0
    embed_freqDomainNoise=0

    augment_chance=1.0/6

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
            inputImage=torchvision.transforms.functional.rotate(inputImage,embed_rotAmount*90)
            imageClean=torchvision.transforms.functional.rotate(imageClean,embed_rotAmount*90)
            if(not(inputMask is None)):
                inputMask=torchvision.transforms.functional.rotate(inputMask,embed_rotAmount*90,fill=[1,0,0])

        #Hue shift:
        if(random.uniform(0, 1)<augment_chance):
            embed_hueShift=random.uniform(-1,1)
            inputImage=(inputImage+1)*0.5
            inputImage=torchvision.transforms.functional.adjust_hue(inputImage,embed_hueShift*0.5)
            inputImage=(inputImage*2)-1

        #Range compression:
        if(random.uniform(0, 1)<augment_chance):
            embed_rangeCompress=random.uniform(-1,1)
            inputImage=inputImage*(1.0-abs(embed_rangeCompress))
            inputImage=inputImage+embed_rangeCompress

        #Frequency Domain Noise:
        if(random.uniform(0, 1)<augment_chance):
            #Convert to frequency domain
            fft_tensor = torch.fft.fft2(inputImage)

            embed_freqDomainNoise=random.uniform(0,0.1)

            #Add Gaussian noise to real/imaginary part
            noise_real = 1+(torch.randn_like(fft_tensor.real) * embed_freqDomainNoise)
            noise_imag = 1+(torch.randn_like(fft_tensor.imag) * embed_freqDomainNoise)
            noise = torch.complex(noise_real, noise_imag)
            fft_tensor = fft_tensor * noise

            #Convert back to spatial domain (and clip within range
            inputImage = torch.fft.ifft2(fft_tensor).real
            inputImage = torch.clamp(inputImage, -1.0, 1.0)

    return_embed=[
        embed_xFlip,
        embed_yFlip,
        embed_rotAmount,
        embed_hueShift,
        embed_rangeCompress,
        embed_freqDomainNoise
        ]

    return inputImage,imageClean,inputMask,return_embed


