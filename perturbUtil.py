import torch
import torch.nn.functional as F


#TODO:
#	wrote code to work in range 0-255 (its what the instructions want) but rest of project is in 0-1
#		possibly add helper function to move between the two, or swap over to 0-1
#	grad-aware round? (probably overkill)
#	preview the output images to make sure its correct


#For a) Gaussian pixel noise
def gaussian_noise(input_image: torch.Tensor, std: float) -> torch.Tensor:
	"""
	Adds gaussian noise to input images (0-255).
	(clips output within range)

	Args:
		input_image (Tensor): Input input_image of size=(batch, *, height, width) in range 0-255
		std (float): Standard deviation of the noise

	Returns:
		(Tensor): Noisy input_image
	"""

	#Add noise
	noisy_image = input_image + torch.randn_like(input_image) * std

	#Clamp within range
	noisy_image = torch.clamp(noisy_image, 0.0, 255.0)

	# Round to nearest integer
	noisy_image = torch.round(noisy_image)

	return noisy_image



#For b) Gaussian blurring
def gaussian_blur(input_image: torch.Tensor, num_iters: int) -> torch.Tensor:
	"""
	Adds gaussian blur to the input images (0-255).

	Args:
		input_image (Tensor): Input tensor of size (batch, 3, height, width)
		num_iters (int): Number of times the blur is applied

	Returns:
		(Tensor): Blurred image
	"""
	#If no blur needed
	if (num_iters<=0):
		return input_image

	#Create the Gaussian kernel
	kernel = torch.tensor([
		[1.0, 2.0, 1.0],
		[2.0, 4.0, 2.0],
		[1.0, 2.0, 1.0]
	], dtype=input_image.dtype, device=input_image.device) / 16.0

	#TODO:
	#	think ive done this correct but make sure later
	#resize kernel for depth-wise conv (groups=3)
	#	shape: (out_channels, in_channels_per_group, kernel_height, kernel_width)
	kernel = kernel.view(1, 1, 3, 3)#(1, 1, 3, 3)
	kernel = kernel.repeat(3, 1, 1, 1)#(3, 1, 3, 3)

	#Apply blur
	for i in range(num_iters):
		input_image = F.conv2d(input_image, kernel, padding=1, groups=3)

	return input_image


#For c) Image Contrast Increase and d) Image Contrast Decrease
def adjust_contrast(input_image: torch.Tensor, contrast_factor: float) -> torch.Tensor:
	"""
	Adjusts the contrast of the input images (0-255).
	(clips output within range)

	Args:
		input_image (Tensor): Input tensor of size (batch, *, height, width) with values in 0-255.
		contrast_factor (float): Number to scale each pixel by.

	Returns:
		(Tensor): Image with its contrast adjusted.
	"""

	#Contrast adjustment
	adjusted = input_image * contrast_factor

	#Clamp to range and round
	adjusted = torch.round(torch.clamp(adjusted, 0.0, 255.0))

	return adjusted



#For e) Image Brightness Increase and f) Image Brightness Decrease
def adjust_brightness(input_image: torch.Tensor, delta: float) -> torch.Tensor:
	"""
	Adjust the brightness of the input images (0-255) by some offset.
	(clips output within range)

	Args:
		input_image (Tensor): Input tensor of size (batch, *, height, width)
		delta (float): Value to add to each pixel

	Returns:
		(Tensor): Image with its brightness adjusted.
	"""

	#Add delta to each pixel
	adjusted = input_image+delta

	#Clamp to range and round
	adjusted = torch.round(torch.clamp(adjusted, 0.0, 255.0))

	return adjusted



#For g) Occlusion of the Image Increase
def occlude_images(input_image: torch.Tensor, size: int) -> torch.Tensor:
	"""
	Overwrites a random square region of each input image (0-255) with black

	Args:
		images (Tensor): Input image of size (batch, *, height, width)
		size (int): width/height of the square

	Returns:
		(Tensor): Occluded images
	"""
	#Make copy to avoid in-place operations
	input_image = input_image+0

	#Get dims
	B=input_image.size(0)
	W=input_image.size(3)
	H=input_image.size(2)

	#Get base coords for each square
	x_start = torch.randint(0, max(H-size,0)+1, size=[B], device=input_image.device)
	y_start = torch.randint(0, max(W-size,0)+1, size=[B], device=input_image.device)

	#TODO: would be nice if this was fully batched
	#Add squares to each image
	for i in range(B):
		x = x_start[i]
		y = y_start[i]

		#To avoid out-of-bounds
		x_end = min(x + size, H)
		y_end = min(y + size, W)

		#Add squares
		input_image[i, :, x:x_end, y:y_end]=0

	return input_image



#For h) Salt and Pepper Noise
def salt_pepper(input_image: torch.Tensor, prob: float) -> torch.Tensor:
	"""
	Adds salt&pepper noise to input images (0-255)

	Args:
		input_image (Tensor): Input image of shape (batch_size, *, height, width) in range 0-255
		prob (float): Probability of a pixel getting replaced.

	Returns:
		(Tensor): Noisy image
	"""

	#mask for which pixels to replace
	mask = torch.rand(input_image.size(0), 1, input_image.size(2), input_image.size(3), device=input_image.device)
	mask = (mask < prob).to(torch.float32)

	#generate noise
	noise = torch.round(torch.rand_like(input_image))*255

	#Add noise to image
	noisy_image = input_image * (1 - mask) + noise * mask

	#Clip range
	noisy_image=torch.clamp(noisy_image, 0.0, 255.0)

	return noisy_image
