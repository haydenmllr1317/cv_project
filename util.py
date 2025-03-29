
import torch

def logit_to_onehot(input_tensor):
	max_indices = input_tensor.argmax(dim=1, keepdim=True)  #Size=(batch, 1, height, width)
	one_hot = torch.zeros_like(input_tensor)
	one_hot.scatter_(1, max_indices, 1.0)

	return one_hot

def get_run_log_dict():
	runLog={
		"LossTrain":[],	#Training set loss
		"IoU_Train":[],	#Training set IoU metric
		"LossTrain_s":[], #Optimizer step for each recording

		"LossDev":[],	#Dev set loss
		"IoU_Dev":[],	#Dev set IoU metric
		"LossDev_s":[], #Optimizer step for each recording
		}
	return runLog
