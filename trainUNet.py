from models.UNet import UNet
from customDataset import imageLoaderDataset
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

import os
from pathlib import Path

import random

import evalUtil
import util
import json

from safetensors.torch import load_model, save_model
import argparse


#Set the device, gpu if available, cpu otherwise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(
	num_epochs=10,
	batch_size = 16,
	maxSteps=10000,
	lr_max=1e-3,
	lr_drop_multiplier=0.1,
	unet_depth=5
	):
	"""
	Performs a U-Net training run.

	Args:
		output (int): The number of epochs to train for.
		batch_size (int): The training batch size.
		maxSteps (int): The max number of optimizer steps. (only used for lr scheduling)
		lr_max (float): The starting learning rate.
		lr_drop_multiplier (float): How much to drop the learning rate by the end of the lr schedule.
		unet_depth (int): The depth of the u-net.
	"""

	#Create model/optimizer and prepare for training
	model = UNet(dimIn=3, dimOut=3, depth=unet_depth).to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr_max)
	model.train()

	#Create train/dev splits with a 6:1 ratio
	dataPairs_Train=util.get_files_in_folder("Dataset/TrainVal/color")
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


	#Set minimum learning rate
	lr_min=lr_max*lr_drop_multiplier

	#Train set dataset/loader
	train_dataset = imageLoaderDataset(dataPairs_Train)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

	#Dev set dataset/loader
	dev_dataset = imageLoaderDataset(dataPairs_Dev)
	dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	dev_iter = iter(dev_loader) #Iterator so it can work inside the main train loop

	#Creates a new run log
	runLog=util.get_run_log_dict()

	globalOptimStep=0

	loss_fn=torch.nn.CrossEntropyLoss()

	#For each epoch
	for epoch in range(num_epochs):

		running_loss = 0

		for batch_idx, (inputImage,imageClean,targetMask) in enumerate(train_loader):

			#Move data to device
			inputImage=inputImage.to(device)
			targetMask=targetMask.to(device)

			#forward
			outputs = model(inputImage,logits=True)
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
					#Switch model to eval mode
					model.eval()

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

					#forward pass
					outputs_dev = model(inputImage_dev, logits=True)
					target_indices_dev = torch.argmax(targetMask_dev, dim=1)
					dev_loss = loss_fn(outputs_dev, target_indices_dev)

					#Switch model back to train mode
					model.train()

					#Metrics:
					IoU_train_set=evalUtil.get_IoU(util.logit_to_onehot(outputs),targetMask)
					IoU_dev_set=evalUtil.get_IoU(util.logit_to_onehot(outputs_dev),targetMask_dev)

					#Record run metrics (train):
					runLog["LossTrain"].append(loss.item())
					runLog["IoU_Train"].append(IoU_train_set.item())
					runLog["LossTrain_s"].append(globalOptimStep)

					#Record run metrics (dev):
					runLog["LossDev"].append(dev_loss.item())
					runLog["IoU_Dev"].append(IoU_dev_set.item())
					runLog["LossDev_s"].append(globalOptimStep)

			globalOptimStep+=1

			#Update learning rate
			if(globalOptimStep>maxSteps):
				newLr=lr_min
			else:
				#Starts at 1, ends at 0
				lr_delta=(1-(globalOptimStep/maxSteps))
				#Starts at lr_max, ends at lr_min
				newLr=(lr_max*lr_delta)+(lr_min*(1.0-lr_delta))
			for g in optimizer.param_groups:
				g['lr'] = newLr

			running_loss += loss.item()

		#Save log file and checkpoint on every epoch
		os.makedirs("Runs/UNet/Run0",exist_ok=True)
		with open("Runs/UNet/Run0/runLog.json","w") as f:
			json.dump(runLog,f)
		#Save a checkpoint every 2 epochs
		if(epoch%2==0):
			os.makedirs("Runs/UNet/Run0/Checkpoints/",exist_ok=True)
			save_model(model, f"Runs/UNet/Run0/Checkpoints/gs{globalOptimStep}_e{epoch}.safetensors")


		#epoch statistics
		epoch_loss = running_loss / len(train_loader)
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



def main():
	#Create parser
	parser = argparse.ArgumentParser(description='Train U-Net Model')

    #Add arguments and default values
	parser.add_argument('--num_epochs', type=int, default=10,
						help='Number of training epochs (default: 10)')
	parser.add_argument('--batch_size', type=int, default=16,
						help='Training batch size (default: 16)')
	parser.add_argument('--max_steps', type=int, default=10000, dest='maxSteps',
						help='Maximum number of training steps, for lr scheduling (default: 10000)')
	parser.add_argument('--lr_max', type=float, default=1e-3,
						help='Starting learning rate (default: 1e-3)')
	parser.add_argument('--lr_drop_multiplier', type=float, default=0.1,
						help='How much to drop the learning rate by the end of the lr schedule (default: 0.1)')
	parser.add_argument('--unet_depth', type=int, default=5,
						help='Depth of the U-Net (default: 5)')


	args = parser.parse_args()

    #Do the training run
	train_model(
		num_epochs=args.num_epochs,
		batch_size=args.batch_size,
		maxSteps=args.maxSteps,
		lr_max=args.lr_max,
		lr_drop_multiplier=args.lr_drop_multiplier,
		unet_depth=args.unet_depth
	)

if __name__ == '__main__':
    main()
