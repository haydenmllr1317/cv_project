from models.UNet import UNet
from models.CLIP_Segmenter import ClIP_Segmentation_Model
from customDataset import imageLoaderDataset
import torch
import torch.nn as nn
import torchvision.transforms as tv_t
from torch.utils.data import DataLoader

import os
from pathlib import Path

import clip
import numpy as np

import random

import evalUtil
import util
import json

from safetensors.torch import load_model, save_model

import perturbUtil
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Name of each perturbation
perturbNames=[
	"Gaussian_pixel_noise",
	"Gaussian_blurring",
	"Image_Contrast_Increase",
	"Image_Contrast_Decrease",
	"Image_Brightness_Increase",
	"Image_Brightness_Decrease",
	"Image_Occlusion",
	"Salt_and_Pepper_Noise",
	]

#The settings to test each perturbation with
perturbParamSweeps=[
	[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
	[0,1,2,3,4,5,6,7,8,9],
	[1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15,1.20, 1.25],
	[1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30,0.20, 0.10],
	[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
	[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
	[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
	[0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14,0.16, 0.18]
	]

def test_model(modelType,modelPath):


	#Load model
	if(modelType=="CLIP"):
		model=ClIP_Segmentation_Model(device).to(device)
	else:
		model=UNet().to(device)
	load_model(model,modelPath)

	#Switch to evaluation mode
	model.eval()

	#Set target resolution
	if(modelType=="CLIP"):
		input_resolution = int(model.encoderModel.visual.input_resolution)
	else:
		input_resolution=128

	clip_image_norm=tv_t.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

	#Put toghether test-set data:
	dataPairs_Test=util.get_files_in_folder("Dataset/Test/color")
	for i in range(len(dataPairs_Test)):
		#Labels seem to always be pngs
		labelImageName=Path(dataPairs_Test[i]).stem+".png"
		dataPairs_Test[i]=(dataPairs_Test[i],f"Dataset/Test/label/{labelImageName}")

	#Test set dataset/loader
	test_dataset = imageLoaderDataset(dataPairs_Test,targetRes=input_resolution,skipAugments=True)
	test_loader = DataLoader(test_dataset, batch_size=16)

	#For each perturbation type
	for perturb_index in range(len(perturbNames)):

		#To store the result of each hyperParam setting
		hyperParam_results=[]

		#Sweep over the different settings
		for hyperparam_index in range(len(perturbParamSweeps[perturb_index])):

			dice_score_sum=0
			num_of_samples=0

			for batch_idx, (inputImage,imageClean,targetMask) in enumerate(test_loader):

				#Move data to device
				inputImage=inputImage.to(device)
				targetMask=targetMask.to(device)

				#Bring data to 0-255 range, as required by perturb script
				inputImage=(inputImage+1.0)*127.5

				hyperParam=perturbParamSweeps[perturb_index][hyperparam_index]

				if(perturbNames[perturb_index]=="Gaussian_pixel_noise"):
					inputImage=perturbUtil.gaussian_noise(inputImage,hyperParam)

				elif(perturbNames[perturb_index]=="Gaussian_blurring"):
					inputImage=perturbUtil.gaussian_blur(inputImage,hyperParam)

				elif(perturbNames[perturb_index]=="Image_Contrast_Increase"):
					inputImage=perturbUtil.adjust_contrast(inputImage,hyperParam)

				elif(perturbNames[perturb_index]=="Image_Contrast_Decrease"):
					inputImage=perturbUtil.adjust_contrast(inputImage,hyperParam)

				elif(perturbNames[perturb_index]=="Image_Brightness_Increase"):
					inputImage=perturbUtil.adjust_brightness(inputImage,hyperParam)

				elif(perturbNames[perturb_index]=="Image_Brightness_Decrease"):
					#Note the negative sign
					inputImage=perturbUtil.adjust_brightness(inputImage,-hyperParam)

				elif(perturbNames[perturb_index]=="Image_Occlusion"):
					inputImage=perturbUtil.occlude_images(inputImage,hyperParam)

				elif(perturbNames[perturb_index]=="Salt_and_Pepper_Noise"):
					inputImage=perturbUtil.salt_pepper(inputImage,hyperParam)

				#Bring data bat to -1 to 1 range.
				inputImage=(inputImage/127.5)-1.0

				#If using clip, bring into 0-1 range, and apply normalization
				if(modelType=="CLIP"):
					inputImage=(inputImage+1.0)*0.5
					inputImage=clip_image_norm(inputImage)

				#forward pass
				outputs_dev = model(inputImage, logits=True)
				target_indices_dev = torch.argmax(targetMask, dim=1)

				#Metrics:
				IoU_dev_set=evalUtil.get_dice_coef(util.logit_to_onehot(outputs_dev),targetMask)

				dice_score_sum+=IoU_dev_set.item()*inputImage.size(0)
				num_of_samples+=inputImage.size(0)

			hyperParam_results.append(dice_score_sum/num_of_samples)
		print(perturbNames[perturb_index])
		print(hyperParam_results)

		os.makedirs(f"perturbEval/{modelType}/",exist_ok=True)
		plt.figure(figsize=(10, 6))

		#Set plot settings
		if(perturbNames[perturb_index]=="Gaussian_pixel_noise"):
			plotTitle="Model Robustness Against Gaussian Noise"
			xlabel="Noise standard deviation"
		elif(perturbNames[perturb_index]=="Gaussian_blurring"):
			plotTitle="Model Robustness Against Gaussian Blurring"
			xlabel="Blur Strength\n(Number of times 3x3 filter is applied)"

		elif(perturbNames[perturb_index]=="Image_Contrast_Increase"):
			plotTitle="Model Robustness Against Increasing Contrast"
			xlabel="Pixel value multiplier"

		elif(perturbNames[perturb_index]=="Image_Contrast_Decrease"):
			plotTitle="Model Robustness Against Decreasing Contrast"
			xlabel="Pixel value multiplier"

		elif(perturbNames[perturb_index]=="Image_Brightness_Increase"):
			plotTitle="Model Robustness Against Increasing Brightness"
			xlabel="Offset added to pixels"

		elif(perturbNames[perturb_index]=="Image_Brightness_Decrease"):
			plotTitle="Model Robustness Against Decreasing Brightness"
			xlabel="Offset subtracted from pixels"

		elif(perturbNames[perturb_index]=="Image_Occlusion"):
			plotTitle="Model Robustness Against Box Occlusion"
			xlabel="Box size (pix)"

		elif(perturbNames[perturb_index]=="Salt_and_Pepper_Noise"):
			plotTitle="Model Robustness Against Salt & Pepper Noise"
			xlabel="Percentage of noisy pixels"

		plt.rcParams.update({'font.size': 16})
		plt.plot(perturbParamSweeps[perturb_index], hyperParam_results, marker='o', linestyle='-', label=f"{modelType} Model Dice Score")
		plt.xlabel(xlabel, fontsize=14)

		#Switch x axis to descending order for contrast decrease.
		if(perturbNames[perturb_index]=="Image_Contrast_Decrease"):
			plt.gca().invert_xaxis()
		#Switch x axis to percentages for Salt_and_Pepper_Noise.
		if(perturbNames[perturb_index]=="Salt_and_Pepper_Noise"):
			plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
		plt.ylabel('Dice score', fontsize=14)
		plt.title(plotTitle, fontsize=16)
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		os.makedirs(f"perturbEval/{modelType}/Run0",exist_ok=True)
		plt.savefig(f'perturbEval/{modelType}/{perturbNames[perturb_index]}.png',bbox_inches='tight', pad_inches = 0.05)



with torch.no_grad():
	#For UNet
	#test_model(modelType="UNet",modelPath="Runs/Clip/Run0/Checkpoints/gs3743_e18.safetensors")
	#For clip
	test_model(modelType="CLIP",modelPath="Runs/Clip/Run0/Checkpoints/gs3349_e16.safetensors")

