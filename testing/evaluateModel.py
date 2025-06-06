#DataLoader/Models
from models.UNet import UNet
from models.CLIP_Segmenter import ClIP_Segmentation_Model
from customDataset import imageLoaderDataset
import clip

#Torch/numpy
import torch
import torch.nn as nn
import torchvision.transforms as tv_t
from torch.utils.data import DataLoader
import numpy as np

#File handling
import os
from pathlib import Path
from safetensors.torch import load_model, save_model

#Utils
import random
import helpful.evalUtil as evalUtil
import util
import json
import argparse
import helpful.perturbUtil as perturbUtil

#Plots
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#Set the device, gpu if available, cpu otherwise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(model_type,model_path,target_split="Test"):
    """
    Evaluates the performance of a given model on a dataset split.
    Provides the dice score of the model on the dataset overall, as well as on cats and dogs specifically

    Args:
        model_type (string): The type of model. One of either: ["CLIP","UNet"]
        model_path (string): Path to the models checkpoint
        target_split (string): Dataset split to test on. One of either: ["Train","Validation","Test"]

    """
    #Ensure valid args
    assert model_type in ["CLIP","UNet"]
    assert target_split in ["Train","Validation","Test"]

    #Load model
    if(model_type=="CLIP"):
        model=ClIP_Segmentation_Model(device).to(device)
    else:
        model=UNet(dimIn=3, dimOut=3, depth=5).to(device)
    load_model(model,model_path)

    #Switch to evaluation mode
    model.eval()

    #Set target resolution
    if(model_type=="CLIP"):
        input_resolution = int(model.encoderModel.visual.input_resolution)
    else:
        input_resolution=128

    clip_image_norm=tv_t.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    #Create train/dev splits with a 6:1 ratio
    dataPairs_Train=util.get_files_in_folder("Dataset/TrainVal/color")
    dataPairs_Train.sort()
    random.seed(0)
    random.shuffle(dataPairs_Train)
    dataPairs_Dev=dataPairs_Train[:len(dataPairs_Train)//7]
    dataPairs_Train=dataPairs_Train[len(dataPairs_Train)//7:]

    #Load appropriate data split
    if(target_split=="Train"):
        #Put toghether train-set data:
        for i in range(len(dataPairs_Train)):
            #Labels seem to always be pngs
            labelImageName=Path(dataPairs_Train[i]).stem+".png"
            dataPairs_Train[i]=(dataPairs_Train[i],f"Dataset/TrainVal/label/{labelImageName}")
        target_set=dataPairs_Train
    elif(target_split=="Validation"):
        #Put toghether validation-set data:
        for i in range(len(dataPairs_Dev)):
            #Labels seem to always be pngs
            labelImageName=Path(dataPairs_Dev[i]).stem+".png"
            dataPairs_Dev[i]=(dataPairs_Dev[i],f"Dataset/TrainVal/label/{labelImageName}")
        target_set=dataPairs_Dev
    elif(target_split=="Test"):
        #Put toghether test-set data:
        dataPairs_Test=util.get_files_in_folder("Dataset/Test/color")
        for i in range(len(dataPairs_Test)):
            #Labels seem to always be pngs
            labelImageName=Path(dataPairs_Test[i]).stem+".png"
            dataPairs_Test[i]=(dataPairs_Test[i],f"Dataset/Test/label/{labelImageName}")
        target_set=dataPairs_Test

    #Split-to-be-tested dataset/loader
    test_dataset = imageLoaderDataset(target_set,targetRes=input_resolution,skipAugments=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    #Required accumilators etc for various metrics
    dice_score_sum=0
    iou_sum=0
    pix_acc_sum=0
    num_of_samples=0
    dice_score_sum_cat=0
    num_of_samples_cat=0
    dice_score_sum_dog=0
    num_of_samples_dog=0

    for batch_idx, (inputImage,imageClean,targetMask) in enumerate(test_loader):

        #Move data to device
        inputImage=inputImage.to(device)
        targetMask=targetMask.to(device)


        #If using clip, bring into 0-1 range, and apply normalization
        if(model_type=="CLIP"):
            inputImage=(inputImage+1.0)*0.5
            inputImage=clip_image_norm(inputImage)

        #forward pass
        outputs_dev = model(inputImage, logits=True)
        target_indices_dev = torch.argmax(targetMask, dim=1)

        #Calculate batch metrics:
        dice_score=evalUtil.get_dice_coef(util.logit_to_onehot(outputs_dev),targetMask)
        iou=evalUtil.get_IoU(util.logit_to_onehot(outputs_dev),targetMask)
        pix_acc=evalUtil.get_pixel_acc(util.logit_to_onehot(outputs_dev),targetMask)

        #Update overall metrics
        dice_score_sum+=dice_score.item()*inputImage.size(0)
        iou_sum+=iou.item()*inputImage.size(0)
        pix_acc_sum+=pix_acc.item()*inputImage.size(0)
        num_of_samples+=inputImage.size(0)

        #Per class metrics:
        for i in range(inputImage.size(0)):
            if(torch.sum(targetMask[i,1]).item()>torch.sum(targetMask[i,2])):
                #cat
                dice_score_sum_cat+=evalUtil.get_dice_coef(util.logit_to_onehot(outputs_dev[i:i+1]),targetMask[i:i+1])
                num_of_samples_cat+=1
            else:
                #dog
                dice_score_sum_dog+=evalUtil.get_dice_coef(util.logit_to_onehot(outputs_dev[i:i+1]),targetMask[i:i+1])
                num_of_samples_dog+=1

    #Print Metrics
    print(f"On {target_split} Set:")
    print(f"\tIoU: {iou_sum/num_of_samples}")
    print(f"\tPixel Accuracy: {pix_acc_sum/num_of_samples}")
    print(f"\tTotal dice score: {dice_score_sum/num_of_samples}")
    print(f"\tcat dice score: {dice_score_sum_cat/num_of_samples_cat}")
    print(f"\tdog dice score: {dice_score_sum_dog/num_of_samples_dog}")



def main():
    #Create parser
    parser = argparse.ArgumentParser(description="Evaluate a machine learning model.")

    #Add arguments and default values
    parser.add_argument('--model_type', type=str, default='UNet',
                        choices=['CLIP', 'UNet'],
                        help='Model type (default: UNet)')
    parser.add_argument('--model_path', type=str, default="",
                        help='Path to model checkpoint (default: None)')
    parser.add_argument('--target_split', type=str, default='Test',
                        choices=['Train', 'Validation', 'Test'],
                        help='Dataset split to evaluate (default: Test)')

    args = parser.parse_args()

    #Run model evaluation
    with torch.no_grad():
        evaluate_model(args.model_type, args.model_path, args.target_split)

if __name__ == "__main__":
    main()
