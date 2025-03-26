import os
import cv2

# file where I check to make sure our validation and test sets weren't
# corrupted at all
# i am simply going to read in the images, check their size, and then write them
# out to another folder



hm_test_color = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/test/color/'
hm_test_label = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/test/label/'

hm_val_color = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/val/color/'
hm_val_label = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/val/label/'

hm_output = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/crap/'

unsorted_test_photos = os.listdir(hm_test_color)
unsorted_test_labels = os.listdir(hm_test_label)
unsorted_val_photos = os.listdir(hm_val_color)
unsorted_val_labels = os.listdir(hm_val_label)
test_photos = sorted(unsorted_test_photos)
test_labels = sorted(unsorted_test_labels)
val_photos = sorted(unsorted_val_photos)
val_labels = sorted(unsorted_val_labels)

if len(test_photos) != len(test_labels): raise ValueError('test size mismatch')
if len(val_photos) != len(val_labels): raise ValueError('val size mismatch')

for i in range(len(test_photos)):
    new_data_path = os.path.join(hm_output, test_photos[i])
    new_labels_path = os.path.join(hm_output, test_labels[i])
    img = cv2.imread(hm_test_color + test_photos[i])
    label = cv2.imread(hm_test_label+ test_labels[i])
    print(str(test_photos[i]))
    print(str(test_labels[i]))
    print(str(img.shape[:2]))
    print(str(label.shape[:2]))
    cv2.imwrite(new_data_path,img)
    cv2.imwrite(new_labels_path,label)

for i in range(len(val_photos)):
    new_data_path = os.path.join(hm_output, val_photos[i])
    new_labels_path = os.path.join(hm_output, val_labels[i])
    img = cv2.imread(hm_val_color + val_photos[i])
    label = cv2.imread(hm_val_label + val_labels[i])
    print(str(val_photos[i]))
    print(str(val_labels[i]))
    print(str(img.shape[:2]))
    print(str(label.shape[:2]))
    cv2.imwrite(new_data_path,img)
    cv2.imwrite(new_labels_path,label)


##YAYAY
# this worked, so all of our val and test data is good