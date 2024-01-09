import os, glob, tqdm, cv2
from scipy import signal
import numpy as np
from natsort import natsorted
from pathlib import Path
import matplotlib.pyplot as plt

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def del_dir_contents(path_to_dir):
    files = glob.glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)

def norm(img):

    img_min = np.min(img)
    img = img-img_min

    img_max = np.max(img)

    if img_max > 0:
        return img/img_max
    else:
        return img

def norm_5_to_95(img):

    img_mean = np.mean(img)
    img_std = np.std(img)

    img2 = img-(img_mean + (2*img_std))

    img_mean2 = np.mean(img2)
    img_std2 = np.std(img2)

    img3 = img2/(np.abs(img_mean2) + np.abs(2*img_std2))
    img3 = (np.clip(img3,-1,1)+1)/2

    return img3

# change this to wheere the datasets is for both conditions in the dataset
dataset_dir = r"Y:\Users\Sam Freitas\WP_imager\analysis\grid_output"
filetype = ".png"

specified_images_in_this_day = natsorted(glob.glob(os.path.join(dataset_dir,'*' + filetype))) 

output_dataset2 = r"C:\Users\LabPC2\Desktop\_SICKO_NN\testing_WPdata"
output_filetype2 = ".png"
# os.makedirs(output_dataset,exist_ok=True)
# del_dir_contents(output_dataset)
os.makedirs(output_dataset2,exist_ok=True)
del_dir_contents(output_dataset2)

# write all the images to the output folder

# kernel = np.ones((25,25),np.uint8)

kernel = (gkern(21)>0.0025).astype(np.uint8)

for j,each_img in enumerate(tqdm.tqdm(specified_images_in_this_day)):
    img = cv2.imread(each_img,-1)

    img = img.astype(np.float32)#/img_max # read into 16 bit, change to float, divide by max value

    img2 = img.copy()
    img2 = cv2.GaussianBlur(img2, ksize=(0, 0), sigmaX=25, borderType=cv2.BORDER_REPLICATE)
    img2 = cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)

    img2 = img-img2

    img2 = norm_5_to_95(img2)
    img2 = (img2*255).astype(np.uint8)

    this_img_name2 = str(j) + output_filetype2
    # cv2.imwrite(os.path.join(output_dataset,this_img_name), img2)
    cv2.imwrite(os.path.join(output_dataset2,this_img_name2), img2)

print('EOF')