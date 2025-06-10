import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights
from monai.losses import PerceptualLoss # Can also be used if the user wants too
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util.arraycrop import crop
from Config_PreProc_Train_Param import CONFIG_Model_Training
cuda_gpu = CONFIG_Model_Training["CUDA"]
device = torch.device(cuda_gpu if torch.cuda.is_available() else "cpu")
from cupyx.scipy.ndimage import rotate, binary_dilation
import cupy as cp
import gc
import math
import glob
import os

dic_PTV_training_36Gy = {} #0 loss
dic_Bladder_training_36Gy = {} #0 loss
dic_Rectum_training_36Gy = {} #0 loss

dic_PTV_training_70Gy = {} #0 loss
dic_Bladder_training_70Gy = {} #0 loss
dic_Rectum_training_70Gy = {} #0 loss

dic_PTV_Val_36Gy = {} #0 loss
dic_Bladder_Val_36Gy = {} #0 loss
dic_Rectum_Val_36Gy = {} #0 loss

dic_PTV_Val_70Gy = {} #0 loss
dic_Bladder_Val_70Gy = {} #0 loss
dic_Rectum_Val_70Gy = {} #0 loss


dic_PTV_test_36Gy = {} #0 loss
dic_CTV_test_36Gy = {} #0 loss
dic_Bladder_test_36Gy = {} #0 loss
dic_Rectum_test_36Gy = {} #0 loss
dic_BODY_test_36Gy = {}


dic_PTV_test_70Gy = {} #0 loss
dic_CTV_test_70Gy = {} #0 loss
dic_Bladder_test_70Gy = {} #0 loss
dic_Rectum_test_70Gy = {} #0 loss
dic_BODY_test_70Gy = {} #0 loss


def OAR_path_mapper_36Gy_training(path, list_of_patients_for_train_or_val):
    list_of_patient = os.listdir(path)

    for patient in list_of_patient[:]:
        list_of_structures_PTV = []
        list_of_structures_Bladder = []
        list_of_structures_Rectum = []
        
        BODY_training_36_structures_PTV = glob.glob(os.path.join(path,f"{patient}","RS_*","PT**_36.0.**"))
        BODY_training_36_structures_Bladder = glob.glob(os.path.join(path,f"{patient}","RS_*","Blad**"))
        BODY_training_36_structures_Rectum = glob.glob(os.path.join(path,f"{patient}","RS_*","Rect**"))    
        
        for structure in BODY_training_36_structures_PTV:
            list_of_structures_PTV.append(structure.split("/")[-1])
        dic_PTV_training_36Gy[f"{patient}"] =  list_of_structures_PTV
        
        for structure in BODY_training_36_structures_Bladder:
            list_of_structures_Bladder.append(structure.split("/")[-1])
        dic_Bladder_training_36Gy[f"{patient}"] =  list_of_structures_Bladder
        
        for structure in BODY_training_36_structures_Rectum:
            list_of_structures_Rectum.append(structure.split("/")[-1])
        dic_Rectum_training_36Gy[f"{patient}"] =  list_of_structures_Rectum  
        
        
    for fraction in list_of_patients_for_train_or_val:
        patient_name = fraction["img"].split("/")[-3]
        structure_name_PTV = dic_PTV_training_36Gy[patient_name]
        structure_name_Bladder = dic_Bladder_training_36Gy[patient_name]
        structure_name_Rectum = dic_Rectum_training_36Gy[patient_name]
        
        fraction["PTV"] = f"{path}{patient_name}/RS_RD/{structure_name_PTV[0]}"
        fraction["Bladder"] = f"{path}{patient_name}/RS_RD/{structure_name_Bladder[0]}"
        fraction["Rectum"] = f"{path}{patient_name}/RS_RD/{structure_name_Rectum[0]}"
    
    return list_of_patients_for_train_or_val

def OAR_path_mapper_70Gy_training(path, list_of_patients_for_train_or_val):
    list_of_patient = os.listdir(path)

    for patient in list_of_patient[:]:
        list_of_structures_PTV = []
        list_of_structures_Bladder = []
        list_of_structures_Rectum = []
        
        BODY_training_70_structures_PTV = glob.glob(os.path.join(path,f"{patient}","RS_*","PT**_70.0.**"))
        BODY_training_70_structures_Bladder = glob.glob(os.path.join(path,f"{patient}","RS_*","Blad**"))
        BODY_training_70_structures_Rectum = glob.glob(os.path.join(path,f"{patient}","RS_*","Rect**"))    
        
        for structure in BODY_training_70_structures_PTV:
            list_of_structures_PTV.append(structure.split("/")[-1])
        dic_PTV_training_70Gy[f"{patient}"] =  list_of_structures_PTV
        
        for structure in BODY_training_70_structures_Bladder:
            list_of_structures_Bladder.append(structure.split("/")[-1])
        dic_Bladder_training_70Gy[f"{patient}"] =  list_of_structures_Bladder
        
        for structure in BODY_training_70_structures_Rectum:
            list_of_structures_Rectum.append(structure.split("/")[-1])
        dic_Rectum_training_70Gy[f"{patient}"] =  list_of_structures_Rectum  
        
        
    for fraction in list_of_patients_for_train_or_val:
        patient_name = fraction["img"].split("/")[-3]
        structure_name_PTV = dic_PTV_training_70Gy[patient_name]
        structure_name_Bladder = dic_Bladder_training_70Gy[patient_name]
        structure_name_Rectum = dic_Rectum_training_70Gy[patient_name]
        
        fraction["PTV"] = f"{path}{patient_name}/RS_RD/{structure_name_PTV[0]}"
        fraction["Bladder"] = f"{path}{patient_name}/RS_RD/{structure_name_Bladder[0]}"
        fraction["Rectum"] = f"{path}{patient_name}/RS_RD/{structure_name_Rectum[0]}"
    
    return list_of_patients_for_train_or_val

def OAR_path_mapper_36Gy_validation(path, list_of_patients_for_train_or_val):
    list_of_patient = os.listdir(path)

    for patient in list_of_patient[:]:
        list_of_structures_PTV = []
        list_of_structures_Bladder = []
        list_of_structures_Rectum = []
        
        validation_36_structures_PTV = glob.glob(os.path.join(path,f"{patient}","RS_*","PT**_36.0.**"))
        validation_36_structures_Bladder = glob.glob(os.path.join(path,f"{patient}","RS_*","Blad**"))
        validation_36_structures_Rectum = glob.glob(os.path.join(path,f"{patient}","RS_*","Rect**"))    
        
        for structure in validation_36_structures_PTV:
            list_of_structures_PTV.append(structure.split("/")[-1])
        dic_PTV_Val_36Gy[f"{patient}"] =  list_of_structures_PTV
        
        for structure in validation_36_structures_Bladder:
            list_of_structures_Bladder.append(structure.split("/")[-1])
        dic_Bladder_Val_36Gy[f"{patient}"] =  list_of_structures_Bladder
        
        for structure in validation_36_structures_Rectum:
            list_of_structures_Rectum.append(structure.split("/")[-1])
        dic_Rectum_Val_36Gy[f"{patient}"] =  list_of_structures_Rectum  
        
        
    for fraction in list_of_patients_for_train_or_val:
        patient_name = fraction["img"].split("/")[-3]
        structure_name_PTV = dic_PTV_Val_36Gy[patient_name]
        structure_name_Bladder = dic_Bladder_Val_36Gy[patient_name]
        structure_name_Rectum = dic_Rectum_Val_36Gy[patient_name]
        
        fraction["PTV"] = f"{path}{patient_name}/RS_RD/{structure_name_PTV[0]}"
        fraction["Bladder"] = f"{path}{patient_name}/RS_RD/{structure_name_Bladder[0]}"
        fraction["Rectum"] = f"{path}{patient_name}/RS_RD/{structure_name_Rectum[0]}"
    
    return list_of_patients_for_train_or_val

def OAR_path_mapper_70Gy_validation(path, list_of_patients_for_train_or_val):
    list_of_patient = os.listdir(path)

    for patient in list_of_patient[:]:
        list_of_structures_PTV = []
        list_of_structures_Bladder = []
        list_of_structures_Rectum = []
        
        validation_70_structures_PTV = glob.glob(os.path.join(path,f"{patient}","RS_*","PT**_70.0.**"))
        validation_70_structures_Bladder = glob.glob(os.path.join(path,f"{patient}","RS_*","Blad**"))
        validation_70_structures_Rectum = glob.glob(os.path.join(path,f"{patient}","RS_*","Rect**"))    
        
        for structure in validation_70_structures_PTV:
            list_of_structures_PTV.append(structure.split("/")[-1])
        dic_PTV_Val_70Gy[f"{patient}"] =  list_of_structures_PTV
        
        for structure in validation_70_structures_Bladder:
            list_of_structures_Bladder.append(structure.split("/")[-1])
        dic_Bladder_Val_70Gy[f"{patient}"] =  list_of_structures_Bladder
        
        for structure in validation_70_structures_Rectum:
            list_of_structures_Rectum.append(structure.split("/")[-1])
        dic_Rectum_Val_70Gy[f"{patient}"] =  list_of_structures_Rectum  
        
        
    for fraction in list_of_patients_for_train_or_val:
        patient_name = fraction["img"].split("/")[-3]
        structure_name_PTV = dic_PTV_Val_70Gy[patient_name]
        structure_name_Bladder = dic_Bladder_Val_70Gy[patient_name]
        structure_name_Rectum = dic_Rectum_Val_70Gy[patient_name]
        
        fraction["PTV"] = f"{path}{patient_name}/RS_RD/{structure_name_PTV[0]}"
        fraction["Bladder"] = f"{path}{patient_name}/RS_RD/{structure_name_Bladder[0]}"
        fraction["Rectum"] = f"{path}{patient_name}/RS_RD/{structure_name_Rectum[0]}"
    
    return list_of_patients_for_train_or_val

def OAR_path_mapper_36Gy_test(path, list_of_patients_for_train_or_val):
    list_of_patient = os.listdir(path)

    for patient in list_of_patient[:]:
        list_of_structures_PTV = []
        list_of_structures_Bladder = []
        list_of_structures_Rectum = []
        list_of_structures_BODY = []
        validation_36_structures_PTV = glob.glob(os.path.join(path,f"{patient}","RS_*","PT**_36.0.**"))
        validation_36_structures_Bladder = glob.glob(os.path.join(path,f"{patient}","RS_*","Blad**"))
        validation_36_structures_Rectum = glob.glob(os.path.join(path,f"{patient}","RS_*","Rect**"))    
        validation_36_structures_BODY = glob.glob(os.path.join(path,f"{patient}","RS_*","BOD**"))    
        
        for structure in validation_36_structures_PTV:
            list_of_structures_PTV.append(structure.split("/")[-1])
        dic_PTV_test_36Gy[f"{patient}"] =  list_of_structures_PTV
        
        for structure in validation_36_structures_Bladder:
            list_of_structures_Bladder.append(structure.split("/")[-1])
        dic_Bladder_test_36Gy[f"{patient}"] =  list_of_structures_Bladder
        
        for structure in validation_36_structures_Rectum:
            list_of_structures_Rectum.append(structure.split("/")[-1])
        dic_Rectum_test_36Gy[f"{patient}"] =  list_of_structures_Rectum

        for structure in validation_36_structures_BODY:
            list_of_structures_BODY.append(structure.split("/")[-1])
        dic_BODY_test_36Gy[f"{patient}"] =  list_of_structures_BODY           
        
    for fraction in list_of_patients_for_train_or_val:
        patient_name = fraction["img"].split("/")[-3]
        structure_name_PTV = dic_PTV_test_36Gy[patient_name]
        structure_name_Bladder = dic_Bladder_test_36Gy[patient_name]
        structure_name_Rectum = dic_Rectum_test_36Gy[patient_name]
        structure_name_BODY = dic_BODY_test_36Gy[patient_name]
        
        fraction["PTV"] = f"{path}{patient_name}/RS_RD/{structure_name_PTV[0]}"
        fraction["Bladder"] = f"{path}{patient_name}/RS_RD/{structure_name_Bladder[0]}"
        fraction["Rectum"] = f"{path}{patient_name}/RS_RD/{structure_name_Rectum[0]}"
        fraction["BODY"] = f"{path}{patient_name}/RS_RD/{structure_name_BODY[0]}"
    
    return list_of_patients_for_train_or_val

def OAR_path_mapper_70Gy_test(path, list_of_patients_for_train_or_val):
    list_of_patient = os.listdir(path)

    for patient in list_of_patient[:]:
        list_of_structures_PTV = []
        list_of_structures_Bladder = []
        list_of_structures_Rectum = []
        list_of_structures_BODY = []
        
        validation_70_structures_PTV = glob.glob(os.path.join(path,f"{patient}","RS_*","PT**_70.0.**"))
        validation_70_structures_Bladder = glob.glob(os.path.join(path,f"{patient}","RS_*","Blad**"))
        validation_70_structures_Rectum = glob.glob(os.path.join(path,f"{patient}","RS_*","Rect**"))    
        validation_70_structures_BODY = glob.glob(os.path.join(path,f"{patient}","RS_*","BOD**"))    
        
        for structure in validation_70_structures_PTV:
            list_of_structures_PTV.append(structure.split("/")[-1])
        dic_PTV_test_70Gy[f"{patient}"] =  list_of_structures_PTV      
        
        for structure in validation_70_structures_Bladder:
            list_of_structures_Bladder.append(structure.split("/")[-1])
        dic_Bladder_test_70Gy[f"{patient}"] =  list_of_structures_Bladder
        
        for structure in validation_70_structures_Rectum:
            list_of_structures_Rectum.append(structure.split("/")[-1])
        dic_Rectum_test_70Gy[f"{patient}"] =  list_of_structures_Rectum  

        for structure in validation_70_structures_BODY:
            list_of_structures_BODY.append(structure.split("/")[-1])
        dic_BODY_test_70Gy[f"{patient}"] =  list_of_structures_BODY          
        
    for fraction in list_of_patients_for_train_or_val:
        patient_name = fraction["img"].split("/")[-3]
        structure_name_PTV = dic_PTV_test_70Gy[patient_name]
        structure_name_Bladder = dic_Bladder_test_70Gy[patient_name]
        structure_name_Rectum = dic_Rectum_test_70Gy[patient_name]
        structure_name_BODY = dic_BODY_test_70Gy[patient_name]
        
        fraction["PTV"] = f"{path}{patient_name}/RS_RD/{structure_name_PTV[0]}"
        fraction["Bladder"] = f"{path}{patient_name}/RS_RD/{structure_name_Bladder[0]}"
        fraction["Rectum"] = f"{path}{patient_name}/RS_RD/{structure_name_Rectum[0]}"
        fraction["BODY"] = f"{path}{patient_name}/RS_RD/{structure_name_BODY[0]}"
    
    return list_of_patients_for_train_or_val

def OAR_path_mapper_ExtenedDataset_test(path, list_of_patients_for_testing):
    list_of_patient = os.listdir(path)

    for patient in list_of_patient[:]:
        list_of_structures_PTV = []
        list_of_structures_Bladder = []
        list_of_structures_Rectum = []
        list_of_structures_BODY = []
        validation_36_structures_PTV = glob.glob(os.path.join(path,f"{patient}","RS_*","PT**_**"))
        #print(f"{patient}: extended_structures_PTV: {validation_36_structures_PTV}")
        validation_36_structures_Bladder = glob.glob(os.path.join(path,f"{patient}","RS_*","Blad**"))
        validation_36_structures_Rectum = glob.glob(os.path.join(path,f"{patient}","RS_*","Rect**"))    
        validation_36_structures_BODY = glob.glob(os.path.join(path,f"{patient}","RS_*","BOD**"))    
        
        for structure in validation_36_structures_PTV:
            list_of_structures_PTV.append(structure.split("/")[-1])
        dic_PTV_test_36Gy[f"{patient}"] =  list_of_structures_PTV
        
        for structure in validation_36_structures_Bladder:
            list_of_structures_Bladder.append(structure.split("/")[-1])
        dic_Bladder_test_36Gy[f"{patient}"] =  list_of_structures_Bladder
        
        for structure in validation_36_structures_Rectum:
            list_of_structures_Rectum.append(structure.split("/")[-1])
        dic_Rectum_test_36Gy[f"{patient}"] =  list_of_structures_Rectum

        for structure in validation_36_structures_BODY:
            list_of_structures_BODY.append(structure.split("/")[-1])
        dic_BODY_test_36Gy[f"{patient}"] =  list_of_structures_BODY
                   

    for fraction in list_of_patients_for_testing:
        patient_name = fraction["img"].split("/")[-3]
        structure_name_PTV = dic_PTV_test_36Gy[patient_name]
        structure_name_Bladder = dic_Bladder_test_36Gy[patient_name]
        structure_name_Rectum = dic_Rectum_test_36Gy[patient_name]
        structure_name_BODY = dic_BODY_test_36Gy[patient_name]
        
        fraction["PTV"] = f"{path}{patient_name}/RS_RD/{structure_name_PTV[0]}"
        fraction["Bladder"] = f"{path}{patient_name}/RS_RD/{structure_name_Bladder[0]}"
        fraction["Rectum"] = f"{path}{patient_name}/RS_RD/{structure_name_Rectum[0]}"
        fraction["BODY"] = f"{path}{patient_name}/RS_RD/{structure_name_BODY[0]}"
    
    return list_of_patients_for_testing

#-----------------------------------------------------------------------------

def patient_organizer(first_list_of_patient_imges, second_list_of_patient_images):
    orderd_dic_of_files = []
    for img in first_list_of_patient_imges:
        for idx in range(len(second_list_of_patient_images)):
            if img.split("/")[-1] == second_list_of_patient_images[idx].split("/")[-1] and img.split("/")[-3] == second_list_of_patient_images[idx].split("/")[-3]:
                orderd_dic_of_files.append({"img":img, "label":second_list_of_patient_images[idx]})
    return orderd_dic_of_files

#-----------------------------------------------------------------------------

def Number_of_wrong_matched_patient_data(orderd_files):
    wrong = []
    for i in orderd_files:
        if i["img"].split("/")[-1] !=  i["label"].split("/")[-1]:
            wrong.append(i["img"].split("/")[-3])
    return len(set(wrong)) 

#--------------------------------------- Accelerated 2D and Psudo3D PL Losses -----------------------------

class Accelerated_PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(Accelerated_PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.layer_indices = layers if layers else [3, 8, 15, 22]
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # generated and target are assumed to be of shape: (1, out_slices, H, W)
        # Remove the batch dimension and treat out_slices as batch for VGG:
        # Permute to (out_slices, 1, H, W) then repeat channels to get 3 channels.
        generated = generated.squeeze(0).unsqueeze(1)  # (out_slices, 1, H, W)
        generated = generated.repeat(1, 3, 1, 1)        # (out_slices, 3, H, W)
        target = target.squeeze(0).unsqueeze(1)
        target = target.repeat(1, 3, 1, 1)
        
        # Process all slices in a batch
        gen_features = self.get_features(generated)
        target_features = self.get_features(target)
        
        total_loss = 0
        for gen_feat, target_feat in zip(gen_features, target_features):
            total_loss += nn.functional.mse_loss(gen_feat, target_feat).to(device)
        return total_loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features

class Accelerated_Psudo3D_PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(Accelerated_Fake3D_PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.layer_indices = layers if layers else [3, 8, 15, 22]
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, generated, target):
        # generated and target are assumed to be of shape: (1, out_slices, H, W)
        # Remove the batch dimension and treat out_slices as batch for VGG:
        # Permute to (out_slices, 1, H, W) then repeat channels to get 3 channels.
        generated = generated.squeeze(0).unsqueeze(1)  # (out_slices, 1, H, W)
        target = target.squeeze(0).unsqueeze(1)
        
        generated_1 = generated.repeat(1, 3, 1, 1)        # (out_slices, 3, H, W)
        target_1 = target.repeat(1, 3, 1, 1)
        
        generated_2 = generated.repeat(1, 3, 1, 1).permute(2,1,0,3)        # (out_slices, 3, H, W)
        target_2 = target.repeat(1, 3, 1, 1).permute(2,1,0,3) 
        
        gen_features_1 = self.get_features(generated_1)
        target_features_1 = self.get_features(target_1)
        
        gen_features_2 = self.get_features(generated_2)
        target_features_2 = self.get_features(target_2)
        
        total_loss_1 = 0
        total_loss_2 = 0
        for gen_feat_1, target_feat_1, gen_feat_2, target_feat_2 in zip(gen_features_1, target_features_1, gen_features_2, target_features_2):
            total_loss_1 += nn.functional.mse_loss(gen_feat_1, target_feat_1).to(device)
            total_loss_2 += nn.functional.mse_loss(gen_feat_2, target_feat_2).to(device)
            
        return total_loss_1 + total_loss_2


    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features

#---------------------------------------- DRR Generator --------------------------
def rotate_volume(ct_array, angle_degrees, rotation_axis):
    """Rotate the volume along the specified axis."""
    # Ensure the input is a CuPy array
    if isinstance(ct_array, torch.Tensor):
        ct_array = cp.asarray(ct_array.cpu().numpy())  # Convert PyTorch → CuPy

    rotated_volume = rotate(ct_array, angle=angle_degrees, axes=rotation_axis, reshape=False, order=1)
    return rotated_volume  # Remains a CuPy array

def numpy_normalizer(image):
    """Normalize the image to the range [0, 1] using CuPy."""
    image = image - cp.min(image)
    image = image / cp.max(image)
    return image

def torch_normalizer(image):
    """Normalize the image to the range [0, 1] using CuPy."""
    image = image - torch.min(image)
    image = image / torch.max(image)
    return image

def generate_drr(ct_array, projection_axis=0):
    """Generate a Digitally Reconstructed Radiograph (DRR)."""
    drr_image = cp.sum(ct_array, axis=projection_axis)
    drr_image_normalized = numpy_normalizer(drr_image)
    return drr_image_normalized  # Remains a CuPy array

def pad_drr(drr_image, target_shape=(128, 128)):
    """Pad DRR images to a fixed size using PyTorch."""
    if isinstance(drr_image, cp.ndarray):
        drr_image = torch.tensor(drr_image.get(), device=device)  # Convert CuPy → PyTorch

    current_height, current_width = drr_image.shape[-2:]  # Extract height and width
    target_height, target_width = target_shape

    pad_height = target_height - current_height
    pad_width = target_width - current_width

    if pad_height < 0 or pad_width < 0:
        raise ValueError("The DRR image is larger than the target shape, cannot pad.")

    padded_drr = torch.nn.functional.pad(
        drr_image.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        (0, 0, pad_height // 2, pad_height - pad_height // 2),  # (left, right, top, bottom)
        mode='constant',
        value=0
    )
    
    return padded_drr.squeeze(0)  # Remove batch dimension

#----------------------------------------- CuPy Accelerated DRR Generator from single volume (CBCT) -------------------------

def DRR_generator_CBCT_DRRs(img_CBCT, angles):
    """Generate DRRs from 3D volume at specified angles."""
    if isinstance(img_CBCT, torch.Tensor):
        img_CBCT = cp.asarray(img_CBCT.detach().cpu().numpy())  # Convert PyTorch → CuPy

    drr_0_CBCT = rotate_volume(img_CBCT, angle_degrees=angles[0], rotation_axis=(2, 3))
    drr_image_0_CBCT = generate_drr(drr_0_CBCT, projection_axis=3)
    drr_image_0_padded_CBCT = pad_drr(drr_image_0_CBCT)
    

    drr_270_CBCT = rotate_volume(img_CBCT, angle_degrees=angles[1], rotation_axis=(1, 2))
    drr_image_270_CBCT = generate_drr(drr_270_CBCT, projection_axis=2)
    drr_image_270_padded_CBCT = pad_drr(drr_image_270_CBCT)
    
    
    return torch.cat((drr_image_0_padded_CBCT, drr_image_270_padded_CBCT), dim=1)  # Return PyTorch tensor

#------------------------------------------ Anatomy Strucures Masks ---------------------------------------

def Anatomy_Mask(PTV, Bladder, Rectum):
    PTV_cp = cp.asarray(PTV.cpu().detach().numpy())
    Bladder_cp = cp.asarray(Bladder.cpu().detach().numpy())
    Rectum_cp = cp.asarray(Rectum.cpu().detach().numpy())
    
    Combined_masks = Bladder_cp + Rectum_cp + PTV_cp
    Dialted_Combined_masks = binary_dilation(Combined_masks, iterations=3, brute_force=True)
    Anatomy_mask = cp.where(Dialted_Combined_masks==0, 0, 1)

    return torch.Tensor(Anatomy_mask).to(device=device)

#---------------------------------------- Image Plotter function ---------------------------------------------------------

def plot_images_in_grid(images, list_of_fig_titles ,name):
                    """
                    Plots images in a grid with titles and saves the figure.

                    Parameters:
                    images (list): List of images to plot.
                    """
                    # Calculate number of rows and columns for the grid
                    num_images = len(images)
                    cols = math.ceil(math.sqrt(num_images))
                    rows = math.ceil(num_images / cols)
                    
                    # Create a figure with subplots
                    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
                    
                    # Flatten axes array for easy iteration
                    axes = axes.flatten()
                    list_of_fig_titles = list_of_fig_titles
                    for idx, (ax, img) in enumerate(zip(axes, images)):
                        ax.imshow(img.detach().cpu().squeeze(), cmap='gray')
                        ax.set_title(list_of_fig_titles[idx])
                        ax.axis('off')
                    
                    # Hide any remaining empty subplots
                    for ax in axes[num_images:]:
                        ax.axis('off')
                    
                    # Adjust layout and save the figure
                    plt.tight_layout()
                    plt.savefig(f"{name}.png")
                    plt.close(fig)
                    gc.collect()

#-------------------------------------------------------------------------------------------------------------------------                                        
#---------------------------------------- Masked Metrics Functions Based on the SynthRAD2023 Framework -------------------
#-------------------------------------------------------------------------------------------------------------------------                                        

def Masked_MAE(predicted, target, mask):
    mask = np.where(mask > 0, 1, 0)
    mae_metric = np.sum(np.abs(predicted*mask - target*mask))/(mask.sum())
    return float(mae_metric)

def Masked_SSIM(predicted, target, mask, data_range):
    predicted_clipped = np.clip(predicted, 0, 1)
    target_clipped = np.clip(target, 0, 1)
    
    if mask is not None:
        #binarize mask to be sure
        mask = np.where(mask>0, 1., 0.)
            
        # Mask gt and pred
        predicted_masked_clipped = np.where(mask==0, 0, predicted_clipped)
        target_masked_clipped = np.where(mask==0, 0, target_clipped)
    
    _, ssim_map  = ssim(predicted_masked_clipped, target_masked_clipped, data_range=data_range, 
                        full=True)
    if mask is not None:
        pad = 3
        ssim_value_masked  = crop(ssim_map, pad)[crop(mask, pad).astype(bool)]
        ssim_value_masked = ssim_value_masked.mean(dtype=np.float64)
        
    return ssim_value_masked

def Masked_PSNR(predicted, target, mask, data_range):
    mask = np.where(mask > 0, 1, 0)
    predicted_clipped = np.clip(predicted, 0, 1)
    target_clipped = np.clip(target, 0, 1)
    
    predicted_clipped_masked = predicted_clipped[mask==1]
    target_clipped_masked = target_clipped[mask==1]

    psnr_metric = psnr(predicted_clipped_masked, target_clipped_masked, data_range=data_range)
    return psnr_metric

def HU_to_Norm(image_HU):
    """Normalize the HU image to [0, 1]."""
    im_min = -1024
    im_max = 2000
    norm_image = (image_HU - im_min) / (im_max - im_min)
    return norm_image, im_min, im_max

def Norm_to_HU(norm_image, im_min, im_max):
    """
    Convert a normalized image back to Hounsfield units (HU) using stored min/max values.
    """

    image_HU = norm_image * (im_max - im_min) + im_min
    return image_HU
