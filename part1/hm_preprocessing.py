import os
import cv2
import numpy as np

hm_input_path = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/Dataset_filtered/TrainVal/color/'
hm_output_path_all = '/Users/haydenmiller/S6_Edinburgh/CV/CVP/proc_aug_data/tv_all/color/'


# so I want to take the color file within trainval, and make it all the same aspect ratio

final_dim = 512

def fit_in_frame(img, final_dim):
    height, width = img.shape[:2] # returns a heigh,width tuple (third entry is # of channels)
    ratio = min(final_dim/height, final_dim/width) # to fill one direction of the screen
    # need to split into case where we are shrinking vs expanding an image
    new_w = int(width*ratio)
    new_h = int(height*ratio)
    if height < final_dim and width < final_dim:
        scaled = cv2.resize(img,(new_w, new_h), interpolation=cv2.INTER_CUBIC) # annoyingly width goes first and then height
    else:
        # interpolation=cv2.INTER_AREA here used for shrinking images 
        scaled = cv2.resize(img,(new_w, new_h), interpolation=cv2.INTER_AREA) # annoyingly width goes first and then height

    actual_h, actual_w = scaled.shape[:2]
    remaining_w = final_dim - actual_w # extra pixels in width direction to fill
    remaining_h = final_dim - actual_h # extra pixels in high direction to fill

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

    # now we pad the remaining pixels to get to final_dim with black ([0,0,0])
    final = cv2.copyMakeBorder(scaled, left=l_add, right=r_add, bottom=b_add, top=t_add, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    return final

for photo in os.listdir(hm_input_path):
    try:
        total_path = os.path.join(hm_input_path, photo) # gets each photo path
        img = cv2.imread(total_path) # grabs the photo
        output = fit_in_frame(img, final_dim) #runs fit_in_frame
        new_path = os.path.join(hm_output_path_all, photo) #finds new_path
        cv2.imwrite(new_path, output)
    except cv2.error as e:
        print('error found at ' + str(photo))
        print('error message was: ' + e)