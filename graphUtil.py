import matplotlib.pyplot as plt

#TODO:
#	clean plot_trainDev_loss

#Returns a copy of the input list with ema smoothing applied
def ema_smoothed(input_list : list,ema : float=0.1) -> list:
	"""
	Returns a copy of the input list with exponential-moving-average smoothing applied.

	Args:
		input_list (list): List with values to be smoothed
		ema (float): Smoothing factor to be used of the noise

	Returns:
		(list): A copy of the original list with smoothing applied.
	"""

	values_smooth=[]

	last_val=input_list[0]
	for i in range(len(input_list)):
		last_val=(ema*input_list[i])+((1-ema)*last_val)
		values_smooth.append(last_val)

	return values_smooth



#Plots the training and validation loss for a given run
#Note: never used "semilogy", but wiki seems to say its basicaly the way to do exponential y axis.
#	need to look more into it later
def plot_trainDev_loss(runLog,smoothing=0.1):
	plt.figure(figsize=(10, 6))

	#Apply ema smoothing to the loss for cleaner graph
	lossTrain_smooth=ema_smoothed(runLog["LossTrain"],smoothing)
	lossDev_smooth=ema_smoothed(runLog["LossDev"],smoothing)

	#Training loss
	plt.semilogy(runLog["LossTrain_s"],
		lossTrain_smooth,
		label='Training Loss',
		color='red',
		alpha=0.8)

	#Dev loss
	plt.semilogy(runLog["LossDev_s"],
		lossDev_smooth,
		label='Dev Loss',
		color='blue',
		alpha=0.8)

	plt.xlabel('Optimizer Steps')
	plt.ylabel('Loss')
	plt.title('Training vs Dev Loss')
	plt.legend()
	plt.grid(True, which='both', linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.show()

def test():
	import json
	with open("Runs/Clip/Run0.json","r") as f:
		plot_trainDev_loss(json.load(f))


test()
