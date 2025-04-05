import os
import torch
import cv2
import math
import random
import customDataset
from customDataset import imageLoaderDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plot
import torch.optim as opt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# THIS FILE IS THE TESTING FILE FOR THE AUTOENCODER FROM TASK 2B

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# here is the self-supervised autoencoder structure used for testing
# we are only using the encoder portion of this model
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.MaxPool2d(2, stride=2)

        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=2,stride=2),
            # nn.ReLU() # for first task, just recreating photo so want ReLU, but maybe want Sigmoid for classification later

        )
    def forward(self, _):
        _ = self.encoder(_)
        # print(str(_.size()))
        _ = self.decoder(_)
        return _

# here we load our saved encoder onto this architecture and switch it into
# eval mode
cae = CAE()
encoder_path = ''
encoder = cae.encoder.to(device)
encoder.load_state_dict(torch.load(encoder_path))
encoder.eval()
    
# here is the model structure for the binary pixel-wise segmentation
# which we implement as part of our decoder in the autoencoder model we are
# testing here
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=2)

        )
    def forward(self, _):
        _ = self.decoder(_)
        return _

# here we load our saved model onto this architecture and switch it into
# eval mode
decoder = Decoder()
decoder_path = ''
decoder = decoder.to(device)
decoder.load_state_dict(torch.load(decoder))
decoder.eval()

# this is our binary classification architecture, named "Decider" for the fact
# which we use for pet classification
class Decider(nn.Module):
    def __init__(self):
        super(Decider, self).__init__()
        self.cov = nn.Sequential(

            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)

        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,1),
        )

    def forward(self,_):
        _ = self.cov(_)
        _ = self.mlp(_)
        return _
    
# this method consumes two tensors:
#   output is the output of the decoder model (pixel segmentation)
#   pet is the output of the decder model (list of which pet is in each photo in a batch)
# the model then creates a a three channel tensor by taking all of the pet pixels
# from the decoder and sending them to the right channel depending on which
# pet the decider found in the image. It then concatenates across the channel
# dimension and returns that tensor with pixel indices corresponding to class.
def pet_pixel_prediction(output, pet):
  sig = nn.Sigmoid()
  output = sig(output)
  output = output.cpu().detach()
  pet = pet.cpu().view(pet.shape[0],1,1,1) # fits the size for multiplication
  output = torch.round(output)
  a_cat = output*pet # everything that is a pet and a cat
  a_dog = output*(1-pet) # everything that is a pet and a dog
  bgrnd = 1-a_cat-a_dog # everything that isn't a cat or dog
  return torch.cat([bgrnd, a_cat, a_dog], dim=1)

# this method takes a tensor and maps the indices to colors
# index 0 is background which is mapped to black
# index 1 is cat which is mapped to red
# index 2 is dog which is mapped to green
def test_imgs(tens):
  color_it = np.array([[0,0,0],[255,0,0],[0,255,0]], dtype=np.uint8)
  tens = torch.argmax(tens, dim=0).numpy().astype(int)
  return color_it[tens]

# here we load our saved model onto this architecture and switch it into
# eval mode
decider = Decider()
decider_path = ''
decider = decider.to(device)
decider.load_state_dict(torch.load(decider_path))
decider.eval()


batch_size = 32
test_path = ''
test_label_path = ''

# grab the input photos from test set
test_pairs=customDataset.get_files_in_folder(test_path)
test_pairs.sort()
random.seed(0)
random.shuffle(test_pairs)

#put together data: (add masks to form input output pairs)
for i in range(len(test_pairs)):
  labelImageName=Path(test_pairs[i]).stem+".png"
  test_pairs[i]=(test_pairs[i],os.path.join(test_label_path,labelImageName))

test_dataset = imageLoaderDataset(test_pairs, skipAugments=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=False)

# here are a bunch of variables we initiate
net_PA = 0
n = 0
eps=1e-5
net_intersection = None
net_union = None
net_sum = None
softM = nn.Softmax(dim=1)

encoder.eval()
decoder.eval()
decider.eval()

# this is our test loop:
with torch.no_grad():
    for _,img,label in test_loader:
        n += 1
        img,label = img.to(device),label.to(device)
        latent = encoder(img.squeeze(dim=1))
        pet = decider(latent) # this predicts the pets in each photo
        output = decoder(latent) # here does the pet vs background segmentation
        output = pet_pixel_prediction(output,pet)
        label = label.cpu()
        print(n)

        # now we save the first photo from each batch for inspection
        test_output_path = ''
        test_photo_path = ''
        test_label_path = ''
        photo_output = (test_imgs(output[0]))
        photo_output = (photo_output).astype(np.uint8)
        photo_output = photo_output[..., ::-1]
        cv2.imwrite(test_output_path, photo_output)
        label_output = (test_imgs(label[0]))
        label_output = (label_output).astype(np.uint8)
        label_output = label_output[..., ::-1]
        cv2.imwrite(test_label_path, label_output)
        og_photo = img.squeeze(dim=1)[0].cpu()
        og_photo = og_photo + 1
        og_photo = og_photo*127.5
        og_photo = og_photo.numpy().transpose(1,2,0)
        og_photo = og_photo.astype(np.uint8)
        og_photo = og_photo[..., ::-1]
        cv2.imwrite(test_photo_path, og_photo)

        # now we calculate our accuracy metrics:
        output_choice=torch.argmax(output, dim=1)
        label_choice=torch.argmax(label, dim=1)

        #pixel accuracy:
        mean_ac = torch.mean((torch.abs(output_choice-label_choice)<0.5).to(torch.float32))
        net_PA += mean_ac.item()

        # IoU and Dice
        intersection = torch.sum(output*label,dim=[0,2,3])
        sum = torch.sum(output,dim=[0,2,3]) + torch.sum(label,dim=[0,2,3])
        union = sum - intersection

        if net_sum is None:
            net_sum = sum
        else:
            net_sum += sum

        if net_intersection is None:
            net_intersection = intersection
        else:
            net_intersection += intersection
        if net_union is None:
            net_union = union
        else:
            net_union += union

    # now we calculate our total IoU and Dice Scores
    intersectionOverUnion = (net_intersection + eps) / (net_union + eps)
    intersectionOverUnion = torch.mean(intersectionOverUnion)

    dice = (2*net_intersection + eps) / (net_sum + eps)
    dice = torch.mean(dice)

# And finally, we print our results across the whole testing set!
print('test iou score is ' + str(intersectionOverUnion.item()))
print('test dice score is ' + str(dice.item()))
print('test PA score is ' + str(net_PA/n))