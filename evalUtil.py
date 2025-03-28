import torch

def get_IoU(output, target):
	"""
	Calculates the mean Intersection over Union for a pair of one-hot encoded segmentation masks

	Args:
		output (Tensor): the models output, size=(B, C, H, W)
		target (Tensor): Ground truth mask, size=(B, C, H, W)

	Returns:
		(Tensor): mean intersection over union
	"""

	#to avoid division by 0
	eps=1e-5

	output=torch.round(output)
	target=torch.round(target)

	#(sum of product)
	intersection = torch.sum(output*target,dim=[0,2,3])

	#(sum of mask - intersection)
	union = (torch.sum(output,dim=[0,2,3]) + torch.sum(target,dim=[0,2,3]) - intersection)

	#calculate IoU
	intersectionOverUnion = (intersection + eps) / (union + eps)
	intersectionOverUnion = torch.mean(intersectionOverUnion)
	return intersectionOverUnion

def get_dice_coef(output, target):
	"""
	Calculates the mean dice coefficient for a pair of one-hot encoded segmentation masks

	Args:
		output (Tensor): the models output, size=(B, C, H, W)
		target (Tensor): Ground truth mask, size=(B, C, H, W)

	Returns:
		(Tensor): mean dice score
	"""

	#to avoid division by 0
	eps=1e-5

	output=torch.round(output)
	target=torch.round(target)

	#(sum of product)
	intersection = torch.sum(output * target,dim=[0,2,3])

	#sum predictions/targets
	sum_output = torch.sum(output,dim=[0, 2, 3])
	sum_target = torch.sum(target,dim=[0, 2, 3])

	#get dice coefficient
	diceCoef = (2*intersection+eps) / (sum_output+sum_target+eps)
	diceCoef = torch.mean(diceCoef)
	return diceCoef

def get_pixel_acc(output, target):
	"""
	Calculates the pixel-wise accuracy for a pair of one-hot encoded segmentation masks

	Args:
		output (Tensor): the models output, size=(B, C, H, W)
		target (Tensor): Ground truth mask, size=(B, C, H, W)

	Returns:
		(Tensor): (mean) pixel accuracy
	"""

	#Get the top prediction for output/target
	output=torch.argmax(output, dim=1)
	target=torch.argmax(target, dim=1)

	#Calculate the mean accuracy
	mean_ac = torch.mean((torch.abs(output-target)<0.5).to(torch.float32))
	return mean_ac
