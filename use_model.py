from torchvision.io.image import read_file, read_image, write_jpeg
from torchvision.models import segmentation 
from torchvision import transforms, ops
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, datetime, cv2, glob, tqdm, time, math
from torch.utils.data import Dataset
from natsort import natsorted
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import *
from network.CMUNeXt import CMUNeXt# cmunext, cmunext_s, cmunext_l
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')


def sort_test_imgs_by_name_date(imgs, combine_image_replicates = False):

    # is use combine_image_replcates = True then the 001 002 003 will be treated as the same

    # this is to sort the images by their names then dates
    # will return the batch size of the 

    path_array = []
    for i,this_img in enumerate(imgs):
        path = os.path.normpath(this_img)
        path = path.split(os.sep)
        path = list(path)
        path_array.append(path)
    path_array = np.asarray(path_array) # split the paths and convert them into an array

    A = path_array.copy() # this sorts by the names of the images given, then by anything else in the path (should only be date)
    B = A[natsorted(range(len(A)), key=lambda x: (A[x, -1]))]

    out = [] # dumps sorted arrays back into lists
    for this_path in B:
        out.append(os.path.join(*this_path))

    unique_amount = [] # this is getting the batch size to return (excludes the name of the image) should be by date
    for col_idx in range(path_array.shape[1]-1):
        this_col = path_array[:,col_idx]
        unique_items = natsorted(np.unique(this_col))
        if len(unique_items) > 1:
            unique_amount.append(len(unique_items))
    batch_size = np.max(unique_amount)
    print('Batch size:', batch_size)
    return out, int(batch_size)

def roundup(x, num):
    return int(math.ceil(x / num)) * num
def rounddown(x, num):
    return int(math.floor(x / num)) * num
def norm_torch(img):
    img_min = torch.min(img)
    img = img-img_min

    img_max = torch.max(img)

    if img_max != 0:
        return img/img_max
    else:
        return img
def norm_0_to_95_torch(img):

    img_mean = torch.mean(img)
    img_std = torch.std(img)

    img2 = img-(img_mean + (2*img_std))

    img_mean2 = torch.mean(img2)
    img_std2 = torch.std(img2)

    img3 = img2/(torch.abs(img_mean2) + torch.abs(2*img_std2))

    num = 3 # 5
    img3 = torch.clip(img3,min = img3.min(),max = num)
    img3 = norm_torch(img3)
    return img3

def get_this_model():
    # model = get_model(model_size=50,device=device,freeze_layers=None, weights=False)    
    # model = Unet(in_channels=1, num_classes=1, backbone='convnext_base', activation=torch.nn.GELU).to(device)
    # model = CMUNeXt(input_channel = 1, num_classes = 1).to(device) # base
    # model = CMUNeXt(input_channel = 1, num_classes = 1,dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]).to(device) ## small
    model = CMUNeXt(input_channel=1,num_classes=1,dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]).to(device) ## large
    return model

def detect_hough_circles(img):

    img = norm_0_to_95_torch(img)
    img = np.atleast_3d((img.cpu().squeeze().numpy()*255).astype(np.uint8))
    a_img_c = np.concatenate((img,img,img), axis = -1)
    detected_circles = cv2.HoughCircles(np.squeeze(img),  
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                   param2 = 30, minRadius = 20, maxRadius = 50) 

    if detected_circles is not None:
        detected_circles = np.squeeze(np.uint16(np.around(detected_circles)))
        if detected_circles.shape.__len__() > 1:
            detected_circles = detected_circles[0,:]
        a, b, r = detected_circles[0], detected_circles[1], detected_circles[2] 
        circle_mask = torch.tensor(cv2.circle(np.zeros((img_size,img_size)), (a, b), r, 1, -1)) 
    else:
        circle_mask = torch.zeros((img_size,img_size))

    return circle_mask

# get all the images
# get all the images in a recursive folder
img_file_format = '.png'
path_to_dir = pathlib.Path(r"C:\Users\LabPC2\Desktop\Fluor example data\t1")
all_files = list(map(str,list(path_to_dir.rglob("*"+img_file_format)))) # recursively get all the files from the specified folder and put in a list

# all_test_imgs = natsorted(glob.glob(os.path.join(r'C:\Users\LabPC2\Desktop\_SICKO_NN\SICKO_GOP50_RPA14_N2','*.png')))
all_test_imgs = natsorted(all_files)[0::3]
all_test_imgs, batch_size = sort_test_imgs_by_name_date(all_test_imgs, combine_image_replicates = True)

# specify and load in the model and parts
img_size = 128
model = get_this_model()

model.load_state_dict(torch.load(r"Y:\Users\Sam Freitas\SICKO_NN\trained_weights\model_20231215_161844_training_128_large_L-8417.pt")) #### uncomment this to use a previously trained weights 
model.eval()

# # Calculate available GPU memory
# memory_stats = torch.cuda.memory_stats()
# total_memory = torch.cuda.get_device_properties(0).total_memory
# available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
# available_memory = available_memory / 1024**3
# print(f"Available GPU memory: {available_memory:.2f} GB")
# batch_size = rounddown(available_memory/0.125, 64)
# print(f"Batch size: {batch_size:.2f}")

testing_dataset = SegmentationDataset(all_test_imgs, None, device = device, transforms=None, resize=preprocess(img_size),return_intial_img_aswell = True)
testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, shuffle = False)

# specify outputs
output_path = './output2/'
os.makedirs(output_path,exist_ok=True)
del_dir_contents(output_path)
testing_binary_threshold = 0.5

# run through all and dump to jpg
with torch.no_grad():
    for i, (tinputs, raw_img) in enumerate(tqdm.tqdm(testing_loader)):

        bboxes = []

        if len(tinputs.squeeze().shape) == 2:
            tinputs = tinputs.squeeze().unsqueeze(0).unsqueeze(0)
        elif len(tinputs.squeeze().shape) == 3:
            tinputs = tinputs.squeeze().unsqueeze(1)

        # this detects the circles in the images (wells)
        # then uses that mask to create a more "zoomed in" version of the input image, then preprocesses it for the input
        for j, (each_input,each_input_raw) in enumerate(zip(tinputs,raw_img)):
            circle_mask = detect_hough_circles(each_input).to(device)
            if torch.sum(circle_mask) > 3000:
                large_circle = transforms.Resize((each_input_raw.squeeze().shape[0],each_input_raw.squeeze().shape[1]), antialias=True)(circle_mask.unsqueeze(0))
                bbox = ops.masks_to_boxes(large_circle).squeeze()
                bboxes.append(bbox)
                masked_raw = each_input_raw[:,int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                tinputs[j] = preprocess(img_size)(masked_raw)
            else:
                bboxes.append(0)

        # normalize the inputs for each other (batches)
        tinputs = norm_0_to_95_torch(tinputs)
        # run the images through the model        
        toutputs = model(tinputs)#['out']

        # dump the outputs to a side by side jpg of the input and output masks
        if tinputs.shape[0] > 1:
            for j, (each_input, each_output) in enumerate(zip(tinputs,toutputs)):
                write_outputs_to_images(each_input, each_output, output_path, i = i, j =  '_' + str(j) ,binary_threshold = testing_binary_threshold)
        else:
            write_outputs_to_images(tinputs, toutputs, output_path, i = i, binary_threshold = testing_binary_threshold)






# circle detection using opencv

# a_img = np.atleast_3d((tinputs.cpu().squeeze().numpy()*255).astype(np.uint8))
# a_img_c = np.concatenate((a_img,a_img,a_img), axis = -1)
# detected_circles = cv2.HoughCircles(a_img,  
#                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
#                param2 = 30, minRadius = 1, maxRadius = 40) 
# detected_circles = np.squeeze(np.uint16(np.around(detected_circles)))
# a, b, r = detected_circles[0], detected_circles[1], detected_circles[2] 
# cv2.circle(a_img_c, (a, b), r, (0, 255, 0), 2) 
# plt.imshow(a_img_c)