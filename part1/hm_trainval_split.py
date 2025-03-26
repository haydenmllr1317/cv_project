import os
import random as rm
import cv2
from sklearn.model_selection import train_test_split

# file where we split our training and validation data sets
# our split off the bat is 5/6 for training/both

hm_preprocessed = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/tv_all/color/'
hm_preprocessed_labels = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/tv_all/label/'
hm_output_path_training_data = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/training/color/'
hm_output_path_training_labels = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/training/label/'
hm_output_path_val_data = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/val/color/'
hm_output_path_val_labels = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/val/label/'

unsorted_photos = os.listdir(hm_preprocessed)
unsorted_labels = os.listdir(hm_preprocessed_labels)
photos = sorted(unsorted_photos)
labels = sorted(unsorted_labels)

# for _ in range(len(photos)):
#     real_name = os.path.splitext(photos[_])[0]
#     real_label = os.path.splitext(labels[_])[0]
#     if real_name != real_label:
#         print(str(_))
#         print(real_name)
#         print(real_label)
#         raise ValueError('name mismatch')

if len(photos) != len(labels): raise ValueError('label/data count mismatch')

train_photos, val_photos, train_labels, val_labels = train_test_split(photos,labels, test_size=1/6, random_state=1)

for i in range(len(train_photos)):
    if len(train_photos) != len(train_labels): raise ValueError('training data and labels unequal length')
    new_data_path = os.path.join(hm_output_path_training_data, train_photos[i])
    new_labels_path = os.path.join(hm_output_path_training_labels, train_labels[i])
    img = cv2.imread(hm_preprocessed + train_photos[i])
    label = cv2.imread(hm_preprocessed_labels + train_labels[i])
    if img.shape[:2] != (512,512): raise ValueError('bad shape train photo')
    if label.shape[:2] != (512,512): raise ValueError('bad shape train label!')
    cv2.imwrite(new_data_path,img)
    # if label == None: raise ValueError('the read in didn')
    cv2.imwrite(new_labels_path,label)


for i in range(len(val_photos)):
    if len(val_photos) != len(val_labels): raise ValueError('val data and labels unequal length')
    new_data_path = os.path.join(hm_output_path_val_data, val_photos[i])
    new_labels_path = os.path.join(hm_output_path_val_labels, val_labels[i])
    img = cv2.imread(hm_preprocessed + val_photos[i])
    label = cv2.imread(hm_preprocessed_labels + val_labels[i])
    if img.shape[:2] != (512,512): raise ValueError('bad shape val photo')
    if label.shape[:2] != (512,512): raise ValueError('bad shape val label!')
    cv2.imwrite(new_data_path,img)
    cv2.imwrite(new_labels_path,label)