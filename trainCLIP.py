#Data loaging/models
from models.CLIP_Segmenter import ClIP_Segmentation_Model
from customDataset import imageLoaderDataset

#Numpy/torch
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as tv_t
from torch.utils.data import DataLoader

#File handling
import os
from pathlib import Path
from safetensors.torch import load_model, save_model

#Utils etc
import random
import evalUtil
import util
import json
import argparse


#Set the device, gpu if available, cpu otherwise
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(
    num_epochs=20,
    batch_size = 16,
    maxSteps=3340,
    lr_max=1e-3,
    lr_drop_multiplier=0.1,
    ):
    """
    Performs a CLIP training run.

    Args:
        output (int): The number of epochs to train for.
        batch_size (int): The training batch size.
        maxSteps (int): The max number of optimizer steps. (only used for lr scheduling)
        lr_max (float): The starting learning rate.
        lr_drop_multiplier (float): How much to drop the learning rate by the end of the lr schedule.
    """

    #Create model/optimizer and prepare for training
    clipModel=ClIP_Segmentation_Model(device).to(device)
    clipModel.trainDecoderOnly()
    clipModel.encoderModel.eval()
    optimizer = optim.Adam(clipModel.decoderModel.parameters(), lr=lr_max)

    #Get clips input resolution
    input_resolution = int(clipModel.encoderModel.visual.input_resolution)
    #Initialise the clip image norm
    clip_image_norm=tv_t.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

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
    train_dataset = imageLoaderDataset(dataPairs_Train,targetRes=input_resolution)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    #Dev set dataset/loader
    dev_dataset = imageLoaderDataset(dataPairs_Dev, targetRes=input_resolution)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_iter = iter(dev_loader) #Iterator so it can work inside the main train loop

    #Creates a new run log
    runLog=util.get_run_log_dict()

    globalOptimStep=0

    loss_fn=torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        running_loss = 0

        for batch_idx, (inputImage,imageClean,targetMask) in enumerate(train_loader):

            #Move data to device
            inputImage=inputImage.to(device)
            targetMask=targetMask.to(device)

            #Bring back into 0-1 range, and apply normalization as required by clip
            with torch.no_grad():
                inputImage=(inputImage+1.0)*0.5
                inputImage=clip_image_norm(inputImage)


            #forward
            outputs = clipModel(inputImage,logits=True)
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
                    #Switch decoder to eval mode
                    clipModel.decoderModel.eval()

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

                    #Bring back into 0-1 range, and apply normalization as required by clip
                    with torch.no_grad():
                        inputImage_dev=(inputImage_dev+1.0)*0.5
                        inputImage_dev=clip_image_norm(inputImage_dev)

                    #forward pass
                    outputs_dev = clipModel(inputImage_dev, logits=True)
                    target_indices_dev = torch.argmax(targetMask_dev, dim=1)
                    dev_loss = loss_fn(outputs_dev, target_indices_dev)

                    #Switch decoder back to train mode
                    clipModel.decoderModel.train()

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
        os.makedirs("Runs/Clip/Run0",exist_ok=True)
        with open("Runs/Clip/Run0/runLog.json","w") as f:
            json.dump(runLog,f)
        #Save a checkpoint every 2 epochs
        if(epoch%2==0):
            os.makedirs("Runs/Clip/Run0/Checkpoints/",exist_ok=True)
            save_model(clipModel, f"Runs/Clip/Run0/Checkpoints/gs{globalOptimStep}_e{epoch}.safetensors")


        #epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



def main():
    #Create parser
    parser = argparse.ArgumentParser(description='Train CLIP Model')

    #Add arguments and default values
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size (default: 16)')
    parser.add_argument('--max_steps', type=int, default=3340, dest='maxSteps',
                        help='Maximum number of training steps, for lr scheduling (default: 3340)')
    parser.add_argument('--lr_max', type=float, default=1e-3,
                        help='Starting learning rate (default: 1e-3)')
    parser.add_argument('--lr_drop_multiplier', type=float, default=0.1,
                        help='How much to drop the learning rate by the end of the lr schedule (default: 0.1)')


    args = parser.parse_args()

    #Do the training run
    train_model(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        maxSteps=args.maxSteps,
        lr_max=args.lr_max,
        lr_drop_multiplier=args.lr_drop_multiplier,
    )

if __name__ == '__main__':
    main()

