from UNet import UNet
from customDataset import imageLoaderDataset,custom_collate
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

import os
from pathlib import Path

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
	model = UNet(dimIn=3, dimOut=3, depth=5).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	model.train()

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

	train_dataset = imageLoaderDataset(dataPairs_Train)
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

			#Put toghether output mask batch
			outputStack=[]
			for sampleIdx in range(len(batchData)):
				outputStack.append(batchData[sampleIdx][2])
			outputStack=torch.cat(outputStack,0).to(device)

			#forward
			outputs = model(imageStack,logits=True)
			target_indices = torch.argmax(outputStack, dim=1)
			loss = loss_fn(outputs, target_indices)

			#backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			#Log every 10 optimizer steps
			if(globalOptimStep%10==0):
				print(f'step {globalOptimStep}: Loss: {loss.item():.4f}   ({(batch_idx*batch_size/len(dataPairs_Train))*100:.2f}% +{epoch})')
			globalOptimStep+=1

			running_loss += loss.item()


		#epoch statistics
		epoch_loss = running_loss / len(train_loader)
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



test()

