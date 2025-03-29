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

	train_dataset = imageLoaderDataset(dataPairs_Train,targetRes=input_resolution)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

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

				#with torch.no_grad():
				#	mean_ac = torch.mean((torch.abs(torch.argmax(outputStack, dim=1)-torch.argmax(outputs, dim=1))>0.5).to(torch.float32))
				#	print(mean_ac)
			globalOptimStep+=1

			running_loss += loss.item()


		#epoch statistics
		epoch_loss = running_loss / len(train_loader)
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



test()

