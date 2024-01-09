import os, glob, cv2
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from skimage import morphology, filters

outdir = '_SICKO_NN'
outpath = os.path.join(os.getcwd(),outdir)
os.makedirs(os.path.join(outpath,'training','images'), exist_ok=True) # set up wher everything is going
os.makedirs(os.path.join(outpath,'training','masks'), exist_ok=True) # set up wher everything is going
os.makedirs(os.path.join(outpath,'testing'), exist_ok=True) # set up wher everything is going

training_imgs = natsorted(glob.glob(os.path.join(os.getcwd(),'datastore_for_sorted_data','training','*.jpg'))) # get the images that will be used for training
testing_img = natsorted(glob.glob(os.path.join(os.getcwd(),'datastore_for_sorted_data','testing','*.jpg'))) # get the images that will be used for training
base_imgs = natsorted(glob.glob(os.path.join(os.getcwd(),'SICKO_testing_dataset_png','*.png'))) # base images for training (not masked)
masks_path = os.path.join(os.getcwd(),'datastore_for_sorted_data','output_masks')

print('base:', len(base_imgs), '-- training:', len(training_imgs), '-- testing (not validation):', len(testing_img), '-- sumcheck:', len(training_imgs)+len(testing_img))

for i, img_path in enumerate(training_imgs): # use the index and name of the correct images to parse them

    name = os.path.split(img_path)[-1]
    idx = int(name[:-5])

    img = cv2.imread(base_imgs[idx])
    mask = cv2.imread(os.path.join(masks_path,name))

    img_temp = img[:,:,0]
    img_temp = filters.gaussian(img[:,:,0],sigma = 6)

    mask_base = 1*(mask[:,:,0]>128).astype(np.uint8)
    # mask_noise = (1*morphology.remove_small_objects(img_temp>0.28, min_size=10, connectivity=2)).astype(np.uint8)
    # mask_noise[mask_base>0] = 0
    # mask_end = (np.zeros_like(mask_base)+1) - (mask_noise+mask_base)
    # mask_out = np.concatenate((np.expand_dims(mask_end,2),np.expand_dims(mask_noise,2),np.expand_dims(mask_base,2)),axis=-1)*255

    cv2.imwrite(os.path.join(outpath,'training','images', str(idx) + '.png'),img)
    cv2.imwrite(os.path.join(outpath,'training','masks', str(idx) + '.png'),mask_base*255)

    print(i)


for i, img_path in enumerate(testing_img): # use the index and name of the correct images to parse them

    name = os.path.split(img_path)[-1]
    idx = int(name[:-5])
    img = cv2.imread(base_imgs[idx])
    cv2.imwrite(os.path.join(outpath,'testing', str(i) + '.png'),img)

    print(i)

# print('asdf')
