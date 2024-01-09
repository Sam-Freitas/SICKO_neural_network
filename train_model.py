from torchvision.io.image import read_file, read_image, write_jpeg
from torchvision.models import segmentation 
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, datetime, cv2, glob, tqdm, time
from torch.utils.data import Dataset
from natsort import natsorted
import albumentations as A
from albumentations.pytorch import ToTensorV2
from backbones_unet.model.unet import Unet
from utils import *

from network.CMUNeXt import CMUNeXt# cmunext, cmunext_s, cmunext_l

plt.ioff()
# check cuda or mps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')

def load_data(img_size, number_to_stop_at = 1000000000000):

    all_imgs = read_all_images(natsorted(glob.glob(os.path.join(imgs_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
    all_masks= read_all_images(natsorted(glob.glob(os.path.join(masks_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
    all_test_imgs = read_all_images(natsorted(glob.glob(os.path.join(testing_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))

    return all_imgs, all_masks, all_test_imgs

def add_sythetic_data(X_train, y_train, img_size, number_to_stop_at = 1000000000000,
                        sytheticly_added_worms = False, blank_wells = False, isolated_worms = False):
    print('ADDING SYTHETIC DATA')
    additional_data = []
    additional_masks = []

    if sytheticly_added_worms:
        imgs_path_synthetic = r"C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\images_synthetic"
        masks_path_synthetic = r"C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\masks_synthetic"
        all_imgs = read_all_images(natsorted(glob.glob(os.path.join(imgs_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
        all_masks= read_all_images(natsorted(glob.glob(os.path.join(masks_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
        additional_data = additional_data + all_imgs
        additional_masks = additional_masks + all_masks

    if blank_wells:
        imgs_path_blank_wells = r"C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\images_synthetic_blank"
        masks_path_blank_wells = r"C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\masks_synthetic_blank"
        all_imgs = read_all_images(natsorted(glob.glob(os.path.join(imgs_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
        all_masks= read_all_images(natsorted(glob.glob(os.path.join(masks_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
        additional_data = additional_data + all_imgs
        additional_masks = additional_masks + all_masks

    if isolated_worms:
        imgs_path_isolated_worms = r"C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\isolated_worms"
        masks_path_isolated_worms = r"C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\masks_unmodified"
        all_imgs = read_all_images(natsorted(glob.glob(os.path.join(imgs_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
        all_masks= read_all_images(natsorted(glob.glob(os.path.join(masks_path,'*.png'))), transforms = preprocess(img_size), number_to_stop_at = int(number_to_stop_at))
        additional_data = additional_data + all_imgs
        additional_masks = additional_masks + all_masks


    return X_train + additional_data, y_train + additional_masks

def get_this_model():
    model = CMUNeXt(input_channel=1,num_classes=1,dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]).to(device) ## large
    return model

weights_outputs_path = './trained_weights/'
os.makedirs(weights_outputs_path,exist_ok=True)

training_transforms = A.Compose([
    A.augmentations.crops.transforms.CropAndPad(pad_cval=0,pad_cval_mask=0,keep_size=True,percent=[-0.25, 0.25],p = 0.5), # pad with zeros
    A.augmentations.crops.transforms.CropAndPad(pad_mode=2,keep_size=True,percent=[-0.25, 0.25], p = 0.5), # pad with reflect 
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2()
])

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = './output/'
os.makedirs(output_path,exist_ok=True)
# del_dir_contents(output_path)

img_size = 128

pretrain = False
load_weights = False #False
batch_size = 64 #124
pretrain_epochs = 100
early_stop_patience = 25
training_epochs = 10000
use_h5 = False

imgs_path = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\images_unmodified'
masks_path = r'C:\Users\LabPC2\Desktop\_SICKO_NN\training_unmodified\masks_unmodified'
testing_path = r'C:\Users\LabPC2\Desktop\_SICKO_NN\testing_WPdata2'

model = get_this_model()
loss_fn = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if not pretrain:
    pretrain_epochs = 0

# optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

####### this is now full sending with all the data
########read in all the images and then use the "preprocess" to resize and convert them to grayscale (grayscale is just to make sure theyre single dim)
if not use_h5:
    all_imgs, all_masks, all_test_imgs =  load_data(img_size)

    X_train, X_val, y_train, y_val = train_test_split(all_imgs, all_masks, test_size=0.33, random_state=42)
    X_train, y_train = add_sythetic_data(X_train, y_train, img_size,
                        sytheticly_added_worms = True, blank_wells = True, isolated_worms = True)

    torch.save(X_train,os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","X_train.h5"))
    torch.save(X_val,os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","X_val.h5"))
    torch.save(y_train,os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","y_train.h5"))
    torch.save(y_val,os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","y_val.h5"))
    torch.save(all_test_imgs,os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","all_test_imgs.h5"))
else:
    X_train = torch.load(os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","X_train.h5"))
    X_val = torch.load(os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","X_val.h5"))
    y_train = torch.load(os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","y_train.h5"))
    y_val = torch.load(os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","y_val.h5"))
    all_test_imgs = torch.load(os.path.join(r"C:\Users\LabPC2\Desktop\_SICKO_NN\h5","all_test_imgs.h5"))

training_dataset = SegmentationDataset(X_train,y_train, device = device, transforms=training_transforms)
validation_dataset = SegmentationDataset(X_val,y_val, device = device, transforms=None) ##################whyyyyyyyyyyyyyyyy
testing_dataset = SegmentationDataset(all_test_imgs, None, device = device, transforms=None)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = 1, shuffle = False)

model_path, losses = training_loop(model, EPOCHS = pretrain_epochs+training_epochs, loss_fn = loss_fn, optimizer = optimizer, 
                training_loader = training_loader, validation_loader = validation_loader, testing_loader = testing_loader, 
                output_path = output_path, weights_outputs_path = weights_outputs_path, best_vloss = 10000000, 
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                save_weights = 'last', save_weights_suffix = 'training',
                epoch_number=pretrain_epochs)#'last')


