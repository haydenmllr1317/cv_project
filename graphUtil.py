import matplotlib.pyplot as plt

#Returns a copy of the input list with ema smoothing applied
def ema_smoothed(input_list : list,ema : float=0.1) -> list:
	"""
	Returns a copy of the input list with exponential-moving-average smoothing applied.

	Args:
		input_list (list): List with values to be smoothed
		ema (float): Smoothing factor to be used

	Returns:
		(list): A copy of the original list with smoothing applied.
	"""

	values_smooth=[]

	last_val=input_list[0]
	for i in range(len(input_list)):
		last_val=(ema*input_list[i])+((1-ema)*last_val)
		values_smooth.append(last_val)

	return values_smooth


def plot_trainDev_loss(run_log,smoothing=0.1,output_path=None,model_name=""):
	"""
	Plots the training and validation loss for a given run

	Args:
		run_log (dict): Run log dict object
		smoothing (float): Smoothing factor to be used
		output_path (string): Optional save path. If provided, figure is saved instead of shown.
		model_name (string): Optional name for the model
	"""
	plt.figure(figsize=(10, 6))

	#Apply ema smoothing to the loss for cleaner graph
	lossTrain_smooth=ema_smoothed(run_log["LossTrain"],smoothing)
	lossDev_smooth=ema_smoothed(run_log["LossDev"],smoothing)

	#Training loss
	plt.plot(run_log["LossTrain_s"],
		lossTrain_smooth,
		label='Training-Set Loss',
		color='red',
		alpha=0.8)

	#Dev loss
	plt.plot(run_log["LossDev_s"],
		lossDev_smooth,
		label='Dev-Set Loss',
		color='blue',
		alpha=0.8)

	plt.xlabel('Optimizer Steps')
	plt.ylabel('Loss')
	if(len(model_name)>0):
		model_name=model_name+" "
	#plt.title(f'{model_name}Loss during Training\n(On Augmented Images)')
	plt.title(f'{model_name}Loss during Training')
	plt.legend()
	plt.grid(True, which='both', linestyle='--', alpha=0.5)
	plt.tight_layout()
	if(output_path is None):
		plt.show()
	else:
		plt.rcParams.update({'font.size': 14})
		plt.savefig(output_path,bbox_inches='tight', pad_inches = 0)


def plot_trainDev_IoU(run_log,smoothing=0.1,output_path=None,model_name=""):
	"""
	Plots the training and validation IoU for a given run

	Args:
		run_log (dict): Run log dict object
		smoothing (float): Smoothing factor to be used
		output_path (string): Optional save path. If provided, figure is saved instead of shown.
		model_name (string): Optional name for the model
	"""
	plt.figure(figsize=(10, 6))

	#Apply ema smoothing to the loss for cleaner graph
	lossTrain_smooth=ema_smoothed(run_log["IoU_Train"],smoothing)
	lossDev_smooth=ema_smoothed(run_log["IoU_Dev"],smoothing)

	#Training loss
	plt.plot(run_log["LossTrain_s"],
		lossTrain_smooth,
		label='Training-Set IoU',
		color='red',
		alpha=0.8)

	#Dev loss
	plt.plot(run_log["LossDev_s"],
		lossDev_smooth,
		label='Dev-Set IoU',
		color='blue',
		alpha=0.8)

	plt.xlabel('Optimizer Steps')
	plt.ylabel('IoU')
	if(len(model_name)>0):
		model_name=model_name+" "
	#plt.title(f'{model_name}Intersection over Union score during Training\n(On Augmented Images)')
	plt.title(f'{model_name}IoU score during Training')
	plt.legend()
	plt.grid(True, which='both', linestyle='--', alpha=0.5)
	plt.tight_layout()
	if(output_path is None):
		plt.show()
	else:
		plt.rcParams.update({'font.size': 14})
		plt.savefig(output_path,bbox_inches='tight', pad_inches = 0)


def plot_IoU_loss_dual(runLog_a,runLog_b,smoothing=0.1):
	"""
	Plots the training and validation IoU for 2 given runs

	Args:
		runLog_a (dict): Run A log dict object
		runLog_b (dict): Run B log dict object
		smoothing (float): Smoothing factor to be used
	"""
	plt.figure(figsize=(10, 6))

	#Apply ema smoothing to the loss for cleaner graph
	lossTrain_smooth_a=ema_smoothed(runLog_a["IoU_Train"],smoothing)
	lossDev_smooth_a=ema_smoothed(runLog_a["IoU_Dev"],smoothing)

	#Apply ema smoothing to the loss for cleaner graph
	lossTrain_smooth_b=ema_smoothed(runLog_b["IoU_Train"],smoothing)
	lossDev_smooth_b=ema_smoothed(runLog_b["IoU_Dev"],smoothing)

	#Training loss
	plt.plot(runLog_a["LossTrain_s"],
		lossTrain_smooth_a,
		label='Training IoU',
		color='red',
		alpha=0.8)

	#Dev loss
	plt.plot(runLog_a["LossDev_s"],
		lossDev_smooth_a,
		label='Dev IoU',
		color='blue',
		alpha=0.8)

	#Training loss
	plt.plot(runLog_b["LossTrain_s"],
		lossTrain_smooth_b,
		label='Training IoU',
		color='orange',
		alpha=0.8)

	#Dev loss
	plt.plot(runLog_b["LossDev_s"],
		lossDev_smooth_b,
		label='Dev IoU',
		color='purple',
		alpha=0.8)

	plt.xlabel('Optimizer Steps')
	plt.ylabel('IoU')
	plt.title('Training vs Dev IoU')
	plt.legend()
	plt.grid(True, which='both', linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.show()

def test():
	import json
	#U-Net Section
	if True:
		with open("Runs/UNet/Run0/runLog.json","r") as f:
			plot_trainDev_loss(json.load(f),output_path="Runs/UNet/Run0/loss.png",model_name="U-Net")
		with open("Runs/UNet/Run0/runLog.json","r") as f:
			plot_trainDev_IoU(json.load(f),output_path="Runs/UNet/Run0/IoU.png",model_name="U-Net")

	#CLIP section
	if True:
		with open("Runs/Clip/Run0/runLog.json","r") as f:
			plot_trainDev_loss(json.load(f),output_path="Runs/Clip/Run0/loss.png",model_name="CLIP")
		with open("Runs/Clip/Run0/runLog.json","r") as f:
			plot_trainDev_IoU(json.load(f),output_path="Runs/Clip/Run0/IoU.png",model_name="CLIP")

	#with open("Runs/Clip/Run0/runLog.json","r") as f:
	#	a=json.load(f)
	#with open("Runs/Clip/Run0/runLog_.json","r") as f:
	#	b=json.load(f)
	#plot_trainDev_loss_dual(a,b)


test()
