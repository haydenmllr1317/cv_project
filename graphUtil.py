import matplotlib.pyplot as plt
import argparse
import json

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



"""
Examples:
    Unet:
    python graphUtil.py loss --run_log Runs/UNet/Run0/runLog.json --model_name U-Net
    python graphUtil.py iou --run_log Runs/UNet/Run0/runLog.json --model_name U-Net

    CLIP:
    python graphUtil.py loss --run_log Runs/Clip/Run0/runLog.json --model_name CLIP
    python graphUtil.py iou --run_log Runs/Clip/Run0/runLog.json --model_name CLIP

    Old/new run comparison:
    python graphUtil.py dual --run_log_a Runs/UNet/Run0/runLog.json --run_log_b Runs/UNet/Run0/runLog_old.json

"""
def main():
    #Create "main" parser
    parser = argparse.ArgumentParser(description='Generate training graphs from run logs.')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Type of graph to generate')

    #Arguments present in all 3 graph types
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--run_log', required=True, help='Path to run log pickle file')
    common_parser.add_argument('--smoothing', type=float, default=0.1, help='Smoothing factor (default: 0.1)')
    common_parser.add_argument('--output_path', help='Output path to save figure (optional)')
    common_parser.add_argument('--model_name', default='', help='Model name for plot title (optional)')

    #Loss plot
    loss_parser = subparsers.add_parser('loss', parents=[common_parser], help='Plot training/validation loss')

    #IoU plot parser
    iou_parser = subparsers.add_parser('iou', parents=[common_parser], help='Plot training/validation IoU')

    #Dual IoU comparison plot parser
    dual_parser = subparsers.add_parser('dual', help='Compare two runs')
    dual_parser.add_argument('--run_log_a', required=True, help='Path to first run log pickle file')
    dual_parser.add_argument('--run_log_b', required=True, help='Path to second run log pickle file')
    dual_parser.add_argument('--smoothing', type=float, default=0.1, help='Smoothing factor (default: 0.1)')

    args = parser.parse_args()

    #If single loss/iou graph
    if(args.command in ['loss', 'iou']):
        #Load run log
        with open(args.run_log,"r") as f:
            target_run_log=json.load(f)

        #Generate appropriate graph
        if (args.command == 'loss'):
            plot_trainDev_loss(target_run_log, args.smoothing, args.output_path, args.model_name)
        else:
            plot_trainDev_IoU(target_run_log, args.smoothing, args.output_path, args.model_name)

    elif(args.command == 'dual'): #If dual iou comparison graph
        #Load run data
        with open(args.run_log_a,"r") as f:
            run_log_a=json.load(f)
        with open(args.run_log_b,"r") as f:
            run_log_b=json.load(f)

        #Create grah
        plot_IoU_loss_dual(run_log_a, run_log_b, args.smoothing)

if __name__ == '__main__':
    main()
