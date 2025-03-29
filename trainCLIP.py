from CLIP_Segmenter import ClIP_Segmentation_Model
from customDataset import imageLoaderDataset
import torch.optim as optim
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

#TODO:
#	proper eval loss tracking
#	lr scheduling?
#	ga?
#	better logging
#	console arguments
#	saving
#	cleanup

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Helper func
def get_files_in_folder(folder_path):
	file_list = []
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			full_path = os.path.join(root, file)
			file_list.append(full_path)
	return file_list

def test():

	clipModel=ClIP_Segmentation_Model(device).to(device)
	clipModel.trainDecoderOnly()
	clipModel.encoderModel.eval()

	optimizer = optim.Adam(clipModel.decoderModel.parameters(), lr=0.001)

	input_resolution = int(clipModel.encoderModel.visual.input_resolution)

	clip_image_norm=tv_t.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

	#Create train/dev splits with a 6:1 ratio
	dataPairs_Train=get_files_in_folder("Dataset/TrainVal/color")
	dataPairs_Train.sort()
	random.seed(0)
	random.shuffle(dataPairs_Train)
	dataPairs_Dev=dataPairs_Train[:len(dataPairs_Train)//7]
	dataPairs_Train=dataPairs_Train[len(dataPairs_Train)//7:]

	#put toghether data:
	for i in range(len(dataPairs_Train)):
		#Labels seem to always be pngs
		labelImageName=Path(dataPairs_Train[i]).stem+".png"
		dataPairs_Train[i]=(dataPairs_Train[i],f"Dataset/TrainVal/label/{labelImageName}")
	for i in range(len(dataPairs_Dev)):
		#Labels seem to always be pngs
		labelImageName=Path(dataPairs_Dev[i]).stem+".png"
		dataPairs_Dev[i]=(dataPairs_Dev[i],f"Dataset/TrainVal/label/{labelImageName}")


	num_epochs = 100
	batch_size = 16

	#Train set dataset/loader
	train_dataset = imageLoaderDataset(dataPairs_Train,targetRes=input_resolution)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

	#Dev set dataset/loader
	dev_dataset = imageLoaderDataset(dataPairs_Dev, targetRes=input_resolution)
	dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	dev_iter = iter(dev_loader) #Iterator so it can work inside the main train loop

	#Creates a new run log
	runLog=util.get_run_log_dict()

	globalOptimStep=0

	loss_fn=torch.nn.CrossEntropyLoss()

	for epoch in range(num_epochs):

		running_loss = 0

		for batch_idx, (inputImage,imageClean,targetMask) in enumerate(train_loader):

			#Move data to device
			inputImage=inputImage.to(device)
			targetMask=targetMask.to(device)

			#Bring back into 0-1 range, and apply normalization as required by clip
			with torch.no_grad():
				inputImage=(inputImage+1.0)*0.5
				inputImage=clip_image_norm(inputImage)


			#forward
			outputs = clipModel(inputImage,logits=True)
			target_indices = torch.argmax(targetMask, dim=1)
			loss = loss_fn(outputs, target_indices)

			#backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			#Log every 10 optimizer steps
			if(globalOptimStep%10==0):
				print(f'step {globalOptimStep}: Loss: {loss.item():.4f}   ({(batch_idx*batch_size/len(dataPairs_Train))*100:.2f}% +{epoch})')

				#Calculate dev loss and other metrics
				with torch.no_grad():
					#Switch decoder to eval mode
					clipModel.decoderModel.eval()

					#Handle data loading
					try:
						(inputImage_dev,_,targetMask_dev) = next(dev_iter)
					except StopIteration:
						#Reset iterator if reached the end
						dev_iter = iter(dev_loader)
						(inputImage_dev,_,targetMask_dev) = next(dev_iter)

					#Move data to device
					inputImage_dev=inputImage_dev.to(device)
					targetMask_dev=targetMask_dev.to(device)

					#Bring back into 0-1 range, and apply normalization as required by clip
					with torch.no_grad():
						inputImage_dev=(inputImage_dev+1.0)*0.5
						inputImage_dev=clip_image_norm(inputImage_dev)

					#forward pass
					outputs_dev = clipModel(inputImage_dev, logits=True)
					target_indices_dev = torch.argmax(targetMask_dev, dim=1)
					dev_loss = loss_fn(outputs_dev, target_indices_dev)

					#print(f'Dev Loss: {dev_loss.item():.4f}')

					#Switch decoder back to train mode
					clipModel.decoderModel.train()

					#Metrics:
					IoU_train_set=evalUtil.get_IoU(util.logit_to_onehot(outputs),targetMask)
					IoU_dev_set=evalUtil.get_IoU(util.logit_to_onehot(outputs_dev),targetMask_dev)
					#print(f'Train IoU: {IoU_train_set:.4f}')
					#print(f'Dev IoU: {IoU_dev_set:.4f}')

					#Record run metrics (train):
					runLog["LossTrain"].append(loss.item())
					runLog["IoU_Train"].append(IoU_train_set.item())
					runLog["LossTrain_s"].append(globalOptimStep)

					#Record run metrics (dev):
					runLog["LossDev"].append(dev_loss.item())
					runLog["IoU_Dev"].append(IoU_dev_set.item())
					runLog["LossDev_s"].append(globalOptimStep)

			globalOptimStep+=1

			running_loss += loss.item()

		#Save log file and checkpoint on every epoch
		os.makedirs("Runs/Clip/Run0",exist_ok=True)
		with open("Runs/Clip/Run0/runLog.json","w") as f:
			json.dump(runLog,f)
		os.makedirs("Runs/Clip/Run0/Checkpoints/",exist_ok=True)
		save_model(clipModel, f"Runs/Clip/Run0/Checkpoints/gs{globalOptimStep}_e{epoch}.safetensors")


		#epoch statistics
		epoch_loss = running_loss / len(train_loader)
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



test()

