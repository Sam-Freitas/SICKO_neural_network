# from torchvision.io.image import read_file, read_image, write_jpeg
# from torchvision.models import segmentation 
from torchvision import transforms
# from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, datetime, cv2, glob, tqdm, time, random
# from torch.utils.data import Dataset
from natsort import natsorted
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from backbones_unet.model.unet import Unet
from utils import *

import scipy
from skimage.restoration import estimate_sigma
from skimage.util import random_noise
from skimage.draw import random_shapes

# from network.CMUNeXt import CMUNeXt# cmunext, cmunext_s, cmunext_l
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
        
grayscale_func = Sequential(transforms.Grayscale())

output_img_blank = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\images_synthetic_blank'
output_mask_blank = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\masks_synthetic_blank'
output_worms = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\isolated_worms'
output_img2 = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\images_synthetic'
output_mask2 = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\masks_synthetic'

# os.makedirs(output_img_blank, exist_ok=True)
# os.makedirs(output_mask_blank, exist_ok=True)
# os.makedirs(output_worms, exist_ok=True)
os.makedirs(output_img2, exist_ok=True)
os.makedirs(output_mask2, exist_ok=True)

# del_dir_contents(output_img_blank)
# del_dir_contents(output_mask_blank)
# del_dir_contents(output_worms)
del_dir_contents(output_img2)
del_dir_contents(output_mask2)

all_imgs = natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\training_unmodified\images_unmodified','*.png')))#[:100]
all_masks= natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\training_unmodified\masks_unmodified','*.png')))#[:100]
# all_test_imgs = read_all_images(natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\testing','*.png'))), transforms = preprocess)

N_synthetic_images = len(all_imgs)*2

# ########### this loop creates the isolated worm and the blank mask/associated image
# for i,(this_img_path,this_mask_path) in enumerate(tqdm.tqdm(zip(all_imgs,all_masks), total=len(all_imgs))):

#     img_name = os.path.split(this_img_path)[-1]

#     if '_' not in img_name:

#         this_img = grayscale_func(read_img_custom(this_img_path))
#         this_mask = grayscale_func(read_img_custom(this_mask_path))

#         isolated_worm = ((this_img*(1*(this_mask>0))).numpy().squeeze()*255).astype(np.uint8)
#         cv2.imwrite(os.path.join(output_worms, '_' + img_name), isolated_worm)

#         mask = this_mask.numpy().squeeze()
#         img = this_img.numpy().squeeze()
#         img = (img*255).astype(np.uint8)

#         large_mask = (255*(cv2.blur(mask,(100,100))>0)).astype(np.uint8) # for inpaint around the mask
#         blank_mask = np.zeros_like(large_mask) # for export 

#         estimated_sigma = estimate_sigma(img)
#         noise_whole = random_noise(img, mode = 'gaussian', var = estimated_sigma*8, clip = False, mean = np.mean(img))
#         noise_mask = noise_whole*large_mask

#         dst = cv2.inpaint(img,large_mask,9,cv2.INPAINT_TELEA).astype(np.float64) # cv2.INPAINT_NS #
#         dst[large_mask>0] = ((noise_whole[large_mask>0]) + dst[large_mask>0])/2

#         cv2.imwrite(os.path.join(output_img_blank, '_blank_' + img_name), dst)
#         cv2.imwrite(os.path.join(output_mask_blank, '_blank_' + img_name), blank_mask)

######### the goal is to create N random images 
######### grab random blank image 
######### grad random worm image (and its mask)
######### combine the two 

all_isolated_worms = natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\training_unmodified\isolated_worms','*.png')))
all_blank_imgs = natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\training_unmodified\images_synthetic_blank','*.png')))
all_masks= natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop',r'_SICKO_NN\training_unmodified\masks_unmodified','*.png')))

for i in tqdm.tqdm(range(N_synthetic_images)):

    random_choise = random.random()

    if random_choise < 0.33:

        random_chosen_worm_idx = random.randint(0,len(all_isolated_worms)-1)
        random_chosen_blank_idx = random.randint(0,len(all_blank_imgs)-1)

        this_worm = grayscale_func(read_img_custom(all_isolated_worms[random_chosen_worm_idx]))
        this_mask = grayscale_func(read_img_custom(all_masks[random_chosen_worm_idx])) # use the same index as the worm
        this_img = grayscale_func(read_img_custom(all_blank_imgs[random_chosen_blank_idx]))

        this_worm = torch.tensor(scipy.ndimage.gaussian_filter(this_worm,2))

        isolated_worm = this_worm[this_mask>0]
        isolated_img_spot = this_img[this_mask>0]

        isolated_worm = (isolated_worm - (isolated_worm.mean()-isolated_img_spot.mean()))

        this_img[this_mask>0] = (isolated_worm + isolated_img_spot)/2

        img_name = 'synth_' + str(i) + '.png'

        this_img = this_img.squeeze().numpy()*255
        this_mask = this_mask.squeeze().numpy()*255

        cv2.imwrite(os.path.join(output_img2, '_' + img_name), this_img)
        cv2.imwrite(os.path.join(output_mask2, '_' + img_name), this_mask)

    elif (random_choise > 0.33) and (random_choise < 0.67):

        random_chosen_blank_idx = random.randint(0,len(all_blank_imgs)-1)

        this_img = grayscale_func(read_img_custom(all_blank_imgs[random_chosen_blank_idx]))
        this_mask = grayscale_func(torch.zeros_like(this_img)) # use the same index as the worm

        random_generated_shapes = ((random_shapes((2048, 2048), min_shapes = 2, max_shapes=10,channel_axis=None, allow_overlap = True)[0]))/255
        random_generated_shapes = torch.tensor(1*(random_generated_shapes<1)).to(torch.float)

        this_img = ((random_generated_shapes*this_img.mean()) + this_img)/2

        img_name = 'synth_' + str(i) + '.png'

        this_img = this_img.squeeze().numpy()*255
        this_mask = this_mask.squeeze().numpy()*255

        cv2.imwrite(os.path.join(output_img2, '_' + img_name), this_img)
        cv2.imwrite(os.path.join(output_mask2, '_' + img_name), this_mask)

    else:

        random_chosen_worm_idx = random.randint(0,len(all_isolated_worms)-1)
        random_chosen_blank_idx = random.randint(0,len(all_blank_imgs)-1)

        this_worm = grayscale_func(read_img_custom(all_isolated_worms[random_chosen_worm_idx]))
        this_mask = grayscale_func(read_img_custom(all_masks[random_chosen_worm_idx])) # use the same index as the worm
        this_img = grayscale_func(read_img_custom(all_blank_imgs[random_chosen_blank_idx]))

        this_worm = torch.tensor(scipy.ndimage.gaussian_filter(this_worm,2))

        isolated_worm = this_worm[this_mask>0]
        isolated_img_spot = this_img[this_mask>0]

        isolated_worm = (isolated_worm - (isolated_worm.mean()-isolated_img_spot.mean()))

        this_img[this_mask>0] = (isolated_worm + isolated_img_spot)/2

        # this_img[this_mask>0] = (this_worm[this_mask>0] + this_img[this_mask>0])/2

        random_generated_shapes = ((random_shapes((2048, 2048), min_shapes = 2, max_shapes=10,channel_axis=None, allow_overlap = True)[0]))/255
        random_generated_shapes = torch.tensor(1*(random_generated_shapes<1)).to(torch.float)

        this_img = ((random_generated_shapes*this_img.mean()) + this_img)/2

        img_name = 'synth_' + str(i) + '.png'

        this_img = this_img.squeeze().numpy()*255
        this_mask = this_mask.squeeze().numpy()*255

        cv2.imwrite(os.path.join(output_img2, '_' + img_name), this_img)
        cv2.imwrite(os.path.join(output_mask2, '_' + img_name), this_mask)