from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, datetime, cv2, glob, tqdm, time, random
from natsort import natsorted
from utils import *

from skimage.restoration import estimate_sigma
from skimage.util import random_noise

def norm(img, img_max = None):

    img_min = np.min(img)
    img = img-img_min

    if img_max == None:
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
def imshow(this_image):
    plt.figure()
    plt.imshow(this_image)
    plt.show()
def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path) 
        else:
            yield entry
        
# change this to wheere the datasets is for both conditions in the dataset
testing_dir = r"Y:\Users\Sam Freitas\WP_imager\analysis\grid_output"
filetype = ".png"
specified_images_in_this_day = natsorted(glob.glob(os.path.join(testing_dir,'*' + filetype))) 
output_dataset2 = r"C:\Users\LabPC2\Desktop\_SICKO_NN\testing_WPdata_unmodified"
output_filetype2 = ".png"
os.makedirs(output_dataset2,exist_ok=True)
del_dir_contents(output_dataset2)

print('Making the testing dataset')
for j,each_img in enumerate(tqdm.tqdm(specified_images_in_this_day)):
    img = cv2.imread(each_img,0)

    img2 = (norm(img)*255).astype(np.uint8)

    this_img_name2 = str(j) + output_filetype2
    # cv2.imwrite(os.path.join(output_dataset,this_img_name), img2)
    cv2.imwrite(os.path.join(output_dataset2,this_img_name2), img2)



all_tifs = []
for entry in scantree(r"C:\Users\LabPC2\Desktop\_SICKO_NN\Terasaki Validation SU10"):
    if entry.is_file():
        if '.tif' in entry.path:
            all_tifs.append(entry.path)

specific_tifs = natsorted(all_tifs[0::3])


N_synthetic_images = 1000

output_imgs = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\images_unmodified'
output_masks = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\masks_unmodified'
os.makedirs(output_imgs, exist_ok=True)
os.makedirs(output_masks, exist_ok = True)

transformed_imgs = natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\training\images','*.png')))#[:100]
all_masks= natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\training\masks','*.png')))#[:100]

img_max = 0
maxes = []

########### this loop creates the isolated worm and the blank mask/associated image
for i,(this_img_path,this_mask_path) in enumerate(tqdm.tqdm(zip(transformed_imgs,all_masks), total=len(transformed_imgs))):

    img_name = os.path.split(this_img_path)[-1]

    idx = int(img_name[:-4])

    img = cv2.imread(specific_tifs[idx],-1).astype(np.float32)
    mask = cv2.imread(this_mask_path,0)

    img_export = (norm(img)*255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_imgs, str(idx) + '.png'),img_export)
    cv2.imwrite(os.path.join(output_masks, str(idx) + '.png'),mask)

print('eof')

