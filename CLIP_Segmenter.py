import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from pathlib import Path

import clip
import numpy as np

#TODO:
#	cleanup
#	posiibly a better decoder model (tho current one also seems fine tbh)



#Extracts the image tokens from a clip model, rather than just the class token.
#(bit hacky but couldn't find a cleaner method in the CLIP repo)
#Should match the code in:
#	https://github.com/openai/CLIP/blob/main/clip/model.py
#		clip/model.py -> VisionTransformer -> forward
def extract_CLIP_features(image_batch,model):
	#The model seems to be loaded as f16.
	#Keep note of the input dtype, and temporarely cast it to fp16 for the clip pass.
	dtype_original=image_batch.dtype
	image_batch=image_batch.to(model.dtype)

	vision_model = model.visual
	x = vision_model.conv1(image_batch)#[batch, width, grid, grid]
	batch, width, grid = x.shape[0], x.shape[1], x.shape[2]

	#prepare for transformer
	x = x.reshape(batch, width, -1).permute(0, 2, 1)  # [batch, grid^2, width]
	x = torch.cat([vision_model.class_embedding.expand(batch, 1, -1), x], dim=1)
	x += vision_model.positional_embedding
	x = vision_model.ln_pre(x)

	#run through transformer
	x = x.permute(1, 0, 2)  #[seq_len, batch, width]
	#Cast dtype again, the previous step seems to ruin it
	#(possibly because only some of the clip weights are quantized)
	x=x.to(model.dtype)
	x = vision_model.transformer(x)
	x = x.permute(1, 0, 2)  #[batch, seq_len, width]

	#get clip features (ignore class token)
	patch_embeddings = x[:, 1:, :]  #[batch, grid^2, width]

	#apply final projection
	#TODO: not entirely sure if this is needed, check again later
	patch_embeddings = vision_model.ln_post(patch_embeddings)
	if vision_model.proj is not None:
		patch_embeddings = patch_embeddings @ vision_model.proj

	#un-flatten back into width/height
	output=patch_embeddings.permute(0, 2, 1).view(batch, -1, grid, grid)

	#Cast back into our original dtype
	output=output.to(dtype_original)
	return output

#Model to decode clip image features of size:
#	[batch, 512, 14, 14]
#into segmentation masks of:
#	[batch, outDim, 224, 224]
class SegmentationDecoder(nn.Module):
	def __init__(self,outDim=3):
		super().__init__()

		self.decoder = nn.Sequential(
			#Now: (batch, 512, 14, 14)

			nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(256),
			nn.GELU(),

			#Now: (batch, 256, 28, 28)

			nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(128),
			nn.GELU(),

			#Now: (batch, 128, 56, 56)

			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(64),
			nn.GELU(),

			#Now: (batch, 64, 112, 112)

			nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(32),
			nn.GELU(),

			#Now: (batch, 32, 224, 224)

			nn.Conv2d(32, outDim, kernel_size=1),  #Project to outDim
			nn.Sigmoid()  # Use sigmoid to get values between 0-1
		)

	def forward(self, x):
		return self.decoder(x)

class ClIP_Segmentation_Model(nn.Module):
	def __init__(self,device):
		super().__init__()

		self.encoderModel, _ = clip.load("ViT-B/16", device=device)
		self.decoderModel = SegmentationDecoder()

	#Helper function to enable grads only for decoder model
	def trainDecoderOnly(self):
		for param in self.encoderModel.parameters():
			param.requires_grad = False
		for param in self.decoderModel.parameters():
			param.requires_grad = True

	#Helper function to enable grads for both models
	def trainBoth(self):
		for param in self.encoderModel.parameters():
			param.requires_grad = True
		for param in self.decoderModel.parameters():
			param.requires_grad = True

	#Helper function to disable grads for both models
	def trainNone(self):
		for param in self.encoderModel.parameters():
			param.requires_grad = False
		for param in self.decoderModel.parameters():
			param.requires_grad = False

	def forward(self, x):
		#Get the image features using clip backbone
		x=extract_CLIP_features(x,self.encoderModel)
		#Decode features into segmentation mask, using decoder model
		x=self.decoderModel(x)
		return x


