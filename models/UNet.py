import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO
#	perhaps something different at top layer.
#		(though attention is probably too much for this project)

#Double Convolution block like in the original paper
class DoubleConv(nn.Module):
	def __init__(self, inDim, outDim):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.BatchNorm2d(inDim),
			nn.Conv2d(inDim, outDim, kernel_size=3, padding=1),
			nn.GELU(),
			nn.BatchNorm2d(outDim),
			nn.Conv2d(outDim, outDim, kernel_size=3, padding=1),
			nn.GELU()
		)

	def forward(self, x):
		return self.double_conv(x)

class UpBlock(nn.Module):
	def __init__(self, inDim, outDim):
		super().__init__()
		#Could also try pixel un-shuffle
		self.up = nn.ConvTranspose2d(inDim, outDim, kernel_size=2, stride=2)
		self.conv = DoubleConv(outDim * 2, outDim)

	def forward(self, x, skip):
		x = self.up(x)
		x = torch.cat([x, skip], dim=1)
		return self.conv(x)

class UNet(nn.Module):
	def __init__(self, dimIn=3, dimOut=3, depth=4, baseDim=64):
		super(UNet, self).__init__()
		self.depth = depth

		#Build the encoder modules
		self.encoder = nn.ModuleList()
		currentDim = dimIn
		for i in range(depth):
			newDim = baseDim * (2 ** i)
			self.encoder.append(DoubleConv(currentDim, newDim))
			currentDim = newDim

		#Top layer
		self.topLayer = DoubleConv(
			baseDim * (2 ** (depth - 1)),
			baseDim * (2 ** depth)
		)

		#Build the decoder modules
		self.decoder = nn.ModuleList()
		for i in range(depth):
			block_in = baseDim * (2 ** (depth - i))
			block_out = baseDim * (2 ** (depth - i - 1))
			self.decoder.append(UpBlock(block_in, block_out))

		#Output projection layer
		self.outProj = nn.Conv2d(baseDim, dimOut, kernel_size=1)

	def forward(self, x,logits=False):
		latents = []

		#encode (downscale)
		for down in self.encoder:
			x = down(x)
			latents.append(x)
			x = F.max_pool2d(x, 2)


		x = self.topLayer(x)

		#decode (upscale)
		for i in range(len(self.decoder)):
			skip = latents[-(i+1)]
			x = self.decoder[i](x, skip)

		#out proj
		x = self.outProj(x)
		if(logits):
			return x
		else:
			return F.softmax(x,dim=1)
"""
def test():
    model = UNet(dimIn=3, dimOut=1, depth=6)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).size())

with torch.no_grad():
	test()
"""
