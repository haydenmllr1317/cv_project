
import torch
import os

def logit_to_onehot(input_tensor):
	"""
	Takes in a logit tensor of size (batch,*,height,width) and returns a tensor of the same size,
		which holds an one-hot encoding of the highest logit per pixel.

	Args:
		input_tensor (Tensor): Input logit tensor of size=(batch, *, height, width)

	Returns:
		(Tensor): One-hot encoded tensor representing the index of the highest logit per pixel in the original tensor
	"""
	max_indices = input_tensor.argmax(dim=1, keepdim=True)  #Size=(batch, 1, height, width)
	one_hot = torch.zeros_like(input_tensor)
	one_hot.scatter_(1, max_indices, 1.0)

	return one_hot

def get_run_log_dict():
	"""
	Creates a dictionary representing a "blank"/new training run log.

	Returns:
		(Dict): A python dictionary to be used to track various metrics during training.
	"""
	runLog={
		"LossTrain":[],	#Training set loss
		"IoU_Train":[],	#Training set IoU metric
		"LossTrain_s":[], #Optimizer step for each recording

		"LossDev":[],	#Dev set loss
		"IoU_Dev":[],	#Dev set IoU metric
		"LossDev_s":[], #Optimizer step for each recording
		}
	return runLog

def get_files_in_folder(folder_path):
	"""
	Helper function which takes in a path to a folder and returns a list of paths for each file in said folder
	(Including all sub-directories)

	Args:
		folder_path (str): Path to a directory

	Returns:
		(list): List of string paths for each file under this directory (recursive)
	"""
	file_list = []
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			full_path = os.path.join(root, file)
			file_list.append(full_path)
	return file_list
