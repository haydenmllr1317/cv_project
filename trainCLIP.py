from CLIP_Segmenter import ClIP_Segmentation_Model
from customDataset import imageLoaderDataset,custom_collate
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as tv_t
from torch.utils.data import DataLoader

import os
from pathlib import Path

import clip
import numpy as np

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

	#put toghether data:
	dataPairPaths=get_files_in_folder("Dataset/TrainVal/color")
	for i in range(len(dataPairPaths)):
		#Labels seem to always be pngs
		labelImageName=Path(dataPairPaths[i]).stem+".png"
		dataPairPaths[i]=(dataPairPaths[i],f"Dataset/TrainVal/label/{labelImageName}")


	num_epochs = 100
	batch_size = 16

	train_dataset = imageLoaderDataset(dataPairPaths,targetRes=input_resolution)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=custom_collate,drop_last=True)

	globalOptimStep=0

	loss_fn=torch.nn.CrossEntropyLoss()

	for epoch in range(num_epochs):

		running_loss = 0

		for batch_idx, batchData in enumerate(train_loader):

			#Put toghether input image batch
			imageStack=[]
			for sampleIdx in range(len(batchData)):
				imageStack.append(batchData[sampleIdx][1])
			imageStack=torch.cat(imageStack,0).to(device)
			#Bring back into 0-1 range, and apply normalization as required by clip
			with torch.no_grad():
				imageStack=(imageStack+1.0)*0.5
				imageStack=clip_image_norm(imageStack)

			#Put toghether output mask batch
			outputStack=[]
			for sampleIdx in range(len(batchData)):
				outputStack.append(batchData[sampleIdx][2])
			outputStack=torch.cat(outputStack,0).to(device)


			#forward
			outputs = clipModel(imageStack,logits=True)
			#print(outputs.size())
			#input()
			target_indices = torch.argmax(outputStack, dim=1)
			loss = loss_fn(outputs, target_indices)

			#backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			#Log every 10 optimizer steps
			if(globalOptimStep%10==0):
				print(f'step {globalOptimStep}: Loss: {loss.item():.4f}   ({(batch_idx*batch_size/len(dataPairPaths))*100:.2f}% +{epoch})')

				#with torch.no_grad():
				#	mean_ac = torch.mean((torch.abs(torch.argmax(outputStack, dim=1)-torch.argmax(outputs, dim=1))>0.5).to(torch.float32))
				#	print(mean_ac)
			globalOptimStep+=1

			running_loss += loss.item()


		#epoch statistics
		epoch_loss = running_loss / len(train_loader)
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



test()

