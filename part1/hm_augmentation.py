import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


# file where we augment our training data to increase quantity

# first we will split our trainval data 5/6 1/6
# then with the 5/6 for val we will add 100% more of augmented
# this leaves us with 10/11 ratio

# augmentation Ideas I had:
#  - zooming out (different depths), flipping 90 degrees (different angles),
# gaussian blur (blurry images), and color jitter (different lighting schemes)

hm_color = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/training/color/'
hm_label = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/training/label/'

hm_color_new = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/training/color/added/'
hm_label_new = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/training/label/added/'

unsorted_photos = os.listdir(hm_color)
unsorted_labels = os.listdir(hm_label)
photos = sorted(unsorted_photos)
labels = sorted(unsorted_labels)

if len(photos) != len(labels): raise ValueError('label/data count mismatch')

trans_photos, alter_photos, trans_labels, alter_labels = train_test_split(photos,labels, test_size=1/2, random_state=2)

if len(trans_photos) != len(trans_labels): raise ValueError('training data and labels unequal length')
if len(alter_photos) != len(alter_labels): raise ValueError('training data and labels unequal length')

zoom_photos, rot_photos, zoom_labels, rot_labels = train_test_split(trans_photos,trans_labels, test_size=1/2, random_state=3)
blur_photos, jitter_photos, blur_labels, jitter_labels = train_test_split(alter_photos,alter_labels, test_size=1/2, random_state=3)

if len(zoom_photos) != len(zoom_labels): raise ValueError('training data and labels unequal length')
if len(rot_photos) != len(rot_labels): raise ValueError('training data and labels unequal length')
if len(blur_photos) != len(blur_labels): raise ValueError('training data and labels unequal length')
if len(jitter_photos) != len(jitter_labels): raise ValueError('training data and labels unequal length')

for i in range(len(zoom_photos)):
    new_data_path = os.path.join(hm_color_new, 'aug_' + zoom_photos[i])
    new_labels_path = os.path.join(hm_label_new, 'aug_' + zoom_labels[i])
    img = cv2.imread(hm_color + zoom_photos[i])
    label = cv2.imread(hm_label + zoom_labels[i])
    r = np.random.randint(1,5)
    ratio = r/(r+1)
    img = cv2.resize(img,(int(512*ratio), int(512*ratio)), interpolation=cv2.INTER_AREA) # annoyingly width goes first and then height
    label = cv2.resize(label,(int(512*ratio), int(512*ratio)), interpolation=cv2.INTER_AREA) # annoyingly width goes first and then height
    
    actual_h, actual_w = img.shape[:2]
    remaining_w = 512 - actual_w # extra pixels in width direction to fill
    remaining_h = 512 - actual_h # extra pixels in high direction to fill

    if (remaining_w % 2) == 0:
        l_add = int(remaining_w/2)
        r_add = int(remaining_w/2)
    else:
        l_add = int(remaining_w/2)
        r_add = int(remaining_w/2) + 1

    if (remaining_h % 2) == 0:
        b_add = int(remaining_h/2)
        t_add = int(remaining_h/2)
    else:
        b_add = int(remaining_h/2)
        t_add = int(remaining_h/2) + 1

    img = cv2.copyMakeBorder(img, left=l_add, right=r_add, bottom=b_add, top=t_add, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    label = cv2.copyMakeBorder(label, left=l_add, right=r_add, bottom=b_add, top=t_add, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    cv2.imwrite(new_data_path,img)
    cv2.imwrite(new_labels_path,label)

for i in range(len(rot_photos)):
    new_data_path = os.path.join(hm_color_new, 'aug_' + rot_photos[i])
    new_labels_path = os.path.join(hm_label_new, 'aug_' + rot_labels[i])
    img = cv2.imread(hm_color + rot_photos[i])
    label = cv2.imread(hm_label + rot_labels[i])
    b = np.random.randint(0,2)
    if b == 0:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
    elif b == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: raise ValueError('dont understnad np.randint')
    # print(str(rot_photos[i]))
    # print(str(img.shape[:2]))
    # print(str(rot_labels[i]))
    # print(str(label.shape[:2]))
    cv2.imwrite(new_data_path,img)
    cv2.imwrite(new_labels_path,label)

for i in range(len(blur_photos)):
    if blur_photos[i] == 'added': pass
    else:
        new_data_path = os.path.join(hm_color_new, 'aug_' + blur_photos[i])
        new_labels_path = os.path.join(hm_label_new, 'aug_' + blur_labels[i])
        img = cv2.imread(hm_color + blur_photos[i])
        label = cv2.imread(hm_label + blur_labels[i])
        # print(str(blur_photos[i]))
        # print(str(blur_labels[i]))
        img = cv2.GaussianBlur(img, (7,7), 1.5)
        img = cv2.resize(img,(int(512*(2/3)), int(512*(2/3))), interpolation=cv2.INTER_AREA) # annoyingly width goes first and then height
        label = cv2.resize(label,(int(512*(2/3)), int(512*(2/3))), interpolation=cv2.INTER_AREA) # annoyingly width goes first and then height
        actual_h, actual_w = img.shape[:2]
        remaining_w = 512 - actual_w # extra pixels in width direction to fill
        remaining_h = 512 - actual_h # extra pixels in high direction to fill

        if (remaining_w % 2) == 0:
            l_add = int(remaining_w/2)
            r_add = int(remaining_w/2)
        else:
            l_add = int(remaining_w/2)
            r_add = int(remaining_w/2) + 1

        if (remaining_h % 2) == 0:
            b_add = int(remaining_h/2)
            t_add = int(remaining_h/2)
        else:
            b_add = int(remaining_h/2)
            t_add = int(remaining_h/2) + 1
    
        img = cv2.copyMakeBorder(img, left=l_add, right=r_add, bottom=b_add, top=t_add, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
        label = cv2.copyMakeBorder(label, left=l_add, right=r_add, bottom=b_add, top=t_add, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
        img_h, img_w = img.shape[:2]
        label_h, label_w = label.shape[:2]
        if not (img_h == img_w and img_h == label_h and label_h == label_w):
            raise ValueError('dimension issues')
        theta = np.random.randint(-20,21)
        if int(img_h/2) != 256: raise ValueError('something wrong')
        if int(img_w/2) != 256: raise ValueError('something wrong')
        mat = cv2.getRotationMatrix2D((int(img_h/2), int(img_w/2)), theta, 1)
        img = cv2.warpAffine(img, mat, (512, 512))
        label = cv2.warpAffine(label,mat,(512,512))
        cv2.imwrite(new_data_path,img)
        cv2.imwrite(new_labels_path,label)

for i in range(len(jitter_photos)):
    new_data_path = os.path.join(hm_color_new, 'aug_' + jitter_photos[i])
    new_labels_path = os.path.join(hm_label_new, 'aug_' + jitter_labels[i])
    img = Image.open(hm_color + jitter_photos[i])
    label = cv2.imread(hm_label + jitter_labels[i])
    label = np.fliplr(label)
    cv2.imwrite(new_labels_path,label)
    jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=.05)
    img = jitter(img) #adds jitter to only the image 
    img = np.fliplr(img)
    cv2.imwrite(new_data_path,img)
