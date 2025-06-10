from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch
from Config_PreProc_Train_Param import CONFIG_Model_Training
cuda_gpu = CONFIG_Model_Training["CUDA"]
device = torch.device(cuda_gpu if torch.cuda.is_available() else "cpu")
from monai.data import DataLoader, CacheDataset
from torchmetrics import MeanAbsoluteError
from monai.utils import set_determinism
import matplotlib
matplotlib.use('Agg')
from Utilities import * 
from Collection_of_models import *
import torch.optim as optim
import torch.nn as nn
import numpy as np
import progressbar
import wandb
import time
import glob
import os  

def train():
    print("Device for Training: ", device)
    #torch.set_num_interop_threads(32)
    set_determinism(seed=0)
    
    #---------------------------------------------------------------------------------- Training --------------------------------------------------------------------------------
    patient_folder_path_training_36Gy = r"/home/mluser1/Musti_2D_3D/Crypted_Training_36Gy/"
    patient_folder_path_training_70Gy = r"/home/mluser1/Musti_2D_3D/Crypted_Training_70Gy/"


    Img_CT_traning_36 = glob.glob(os.path.join(patient_folder_path_training_36Gy,"**",'IMGS_*',"*.nii"))
    Label_CBCT_training_36 = glob.glob(os.path.join(patient_folder_path_training_36Gy,"**",'Labels_*',"*.nii"))
    Data_files_training_36 = patient_organizer(first_list_of_patient_imges=Img_CT_traning_36, second_list_of_patient_images=Label_CBCT_training_36)


    Img_CT_traning_70 = glob.glob(os.path.join(patient_folder_path_training_70Gy,"**",'IMGS_*',"*.nii"))
    Label_CBCT_training_70 = glob.glob(os.path.join(patient_folder_path_training_70Gy,"**",'Labels_*',"*.nii"))
    Data_files_training_70 = patient_organizer(first_list_of_patient_imges=Img_CT_traning_70, second_list_of_patient_images=Label_CBCT_training_70)

        
    Data_files_training_36 = OAR_path_mapper_36Gy_training(path=patient_folder_path_training_36Gy, list_of_patients_for_train_or_val=Data_files_training_36) 
    Data_files_training_70 = OAR_path_mapper_70Gy_training(path=patient_folder_path_training_70Gy, list_of_patients_for_train_or_val=Data_files_training_70) 


    training_files = Data_files_training_36 + Data_files_training_70

    #---------------------------------------------------------------------------------- Validation --------------------------------------------------------------------------------
    patient_folder_path_val_36Gy = r"/home/mluser1/Musti_2D_3D/Crypted_Val_36Gy/"
    patient_folder_path_val_70Gy = r"/home/mluser1/Musti_2D_3D/Crypted_Val_70Gy/"


    Img_CT_val_36 = glob.glob(os.path.join(patient_folder_path_val_36Gy,"**",'IMGS_*',"*.nii"))
    Label_CBCT_val_36 = glob.glob(os.path.join(patient_folder_path_val_36Gy,"**",'Labels_*',"*.nii"))
    Data_files_val_36 = patient_organizer(first_list_of_patient_imges=Img_CT_val_36, second_list_of_patient_images=Label_CBCT_val_36)

    Img_CT_val_70 = glob.glob(os.path.join(patient_folder_path_val_70Gy,"**",'IMGS_*',"*.nii"))
    Label_CBCT_val_70 = glob.glob(os.path.join(patient_folder_path_val_70Gy,"**",'Labels_*',"*.nii"))
    Data_files_val_70 = patient_organizer(first_list_of_patient_imges=Img_CT_val_70, second_list_of_patient_images=Label_CBCT_val_70)
    
    
    Data_files_val_36 = OAR_path_mapper_36Gy_validation(path=patient_folder_path_val_36Gy, list_of_patients_for_train_or_val=Data_files_val_36) 
    Data_files_val_70 = OAR_path_mapper_70Gy_validation(path=patient_folder_path_val_70Gy, list_of_patients_for_train_or_val=Data_files_val_70) 
    
    
    val_files = Data_files_val_36 + Data_files_val_70
    
    if Number_of_wrong_matched_patient_data(training_files) == 0 and Number_of_wrong_matched_patient_data(val_files) == 0:
         print("The Data Looks Good, Ready To Train")
    else:
        print("The Data Is Not Correctly Matched")
        
    
    training_files = training_files[:]
    val_files = val_files[:]

    print("Number of Training Samples:", len(training_files))
    print("Number of Validation Samples:", len(val_files))
    #---------------------------------------------------------------------------------- Data Loading --------------------------------------------------------------------------------
    from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureTyped,
    RandRotated,
    RandAffined,
    Resized,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    )
    
    input_image_size = CONFIG_Model_Training["input_image_size"]
    batch_size = CONFIG_Model_Training["batch_size"]
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["img","label", "PTV","Rectum","Bladder"], ensure_channel_first=True),
            EnsureTyped(keys=["img","label", "PTV","Rectum","Bladder"]),
            CenterSpatialCropd(keys=["img","label"], roi_size=(input_image_size,input_image_size,70)),
            Resized(keys=["PTV","Rectum","Bladder"], mode="nearest", spatial_size=(256,256,88)),
            CenterSpatialCropd(keys=["PTV","Rectum","Bladder"], roi_size=(input_image_size,input_image_size,70)),
            RandRotated(keys=["label","PTV","Rectum","Bladder"], prob=0.5, range_y=0.0698/2, range_x=0.0698/2, range_z=0.0698/2, mode=["bilinear","nearest","nearest","nearest"]),
            ScaleIntensityRanged(keys=["img", "label"], a_min=-1024, a_max=2000, b_min=0, b_max=1, clip=True),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img","label", "PTV","Rectum","Bladder"], ensure_channel_first=True),
            EnsureTyped(keys=["img","label", "PTV","Rectum","Bladder"]),
            CenterSpatialCropd(keys=["img","label"], roi_size=(input_image_size,input_image_size,70)),
            Resized(keys=["PTV","Rectum","Bladder"], mode="nearest", spatial_size=(256,256,88)),
            CenterSpatialCropd(keys=["PTV","Rectum","Bladder"], roi_size=(input_image_size,input_image_size,70)),
            ScaleIntensityRanged(keys=["img", "label"], a_min=-1024, a_max=2000, b_min=0, b_max=1, clip=True),
        ]
    )

#-----------------------------------------------------------------------------------------------------
#------------------------------------------- Start Experiment Tracking -------------------------------
#-----------------------------------------------------------------------------------------------------
    alpha = CONFIG_Model_Training["alpha"]
    beta = CONFIG_Model_Training["beta"]
    gamma = CONFIG_Model_Training["gamma"]
    omega = CONFIG_Model_Training["omega"]

    model_names =["Model_6"]

    for model_name in model_names:

        changed_variables =f"{model_name}_OAR_MAEandPL_Light_DRR_3DLoss_beta_{beta}_gamma_{gamma}_omega{omega}"
        Saved_model_name = f"MaskedOAR_ALLModelRuns_DRR_CT_to_CBCT"
        save_images_path = r"/home/mluser1/Musti_2D_3D/Some_Results/"
        Saving_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/"
        Method_name = Saved_model_name.split("_")[0]

        wandb.init(project=Saved_model_name, name= changed_variables ,config={
            "number_of_output_slices": CONFIG_Model_Training["number_of_output_slices"],
            "number_of_projections": len(CONFIG_Model_Training["input_angles_DRRs"]),
            "input_angles_DRRs": CONFIG_Model_Training["input_angles_DRRs"],
            "source_to_detec": CONFIG_Model_Training["source_to_detec"],
            "learning_rate": CONFIG_Model_Training["learning_rate"],
            "val_interval": CONFIG_Model_Training["val_interval"],
            "batch_size": CONFIG_Model_Training["batch_size"],
            "Number of Training Samples": len(training_files),
            "Number of Validation Samples": len(val_files),
            "epochs": CONFIG_Model_Training["max_epochs"],
            "alpha": CONFIG_Model_Training["alpha"],
            "beta": CONFIG_Model_Training["beta"],
            "gamma":  CONFIG_Model_Training["gamma"],
            "omega":  CONFIG_Model_Training["omega"],
            "input_image_size": CONFIG_Model_Training["input_image_size"],
        })


        config = wandb.config

        os.makedirs(os.path.join(save_images_path, Method_name), exist_ok=True)
        os.makedirs(os.path.join(save_images_path, Method_name, changed_variables), exist_ok=True) 

        Saved_model_name = Saved_model_name + changed_variables

        train_ds = CacheDataset( data=training_files, transform=train_transforms,  cache_rate=1, num_workers=None)
        train_loader = DataLoader(train_ds, 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=32,
                                pin_memory=torch.cuda.is_available())


        val_ds = CacheDataset( data=val_files, transform=val_transforms, cache_rate=1, num_workers=None)
        val_loader = DataLoader(val_ds, 
                                batch_size=batch_size,
                                num_workers=32,
                                shuffle=True,
                                pin_memory=torch.cuda.is_available())

        max_epochs = CONFIG_Model_Training["max_epochs"]
        val_interval = CONFIG_Model_Training["val_interval"]
        learning_rate = CONFIG_Model_Training["learning_rate"]
        input_angles_DRRs =  CONFIG_Model_Training["input_angles_DRRs"]
        number_of_output_slices = CONFIG_Model_Training["number_of_output_slices"]
        number_of_projections = len(input_angles_DRRs)
        

        ############################################### Model Import #########################################################
        if model_name == "Model_4":
            Nature_Model = Model_4_Base_Simplified_Light(in_channels_2d= number_of_projections, in_channels_3d=1, out_slices=number_of_output_slices).to(device)
        if model_name == "Model_5":
            Nature_Model = Model_5_Base_2Skips_Simplified_Light(in_channels_2d =number_of_projections, in_channels_3d=1, out_slices=number_of_output_slices).to(device)
        if model_name == "Model_6":
            Nature_Model = Model_6_Base_2Skip_EasyRes_Simplified_Light(in_channels_2d = number_of_projections, in_channels_3d=1, out_slices=number_of_output_slices).to(device)
            #Nature_Model = Model_6_Base_2Skip_EasyRes_Attention_Light(in_channels_2d = number_of_projections, in_channels_3d=1, out_slices=number_of_output_slices).to(device)
        ############################################### Training of Model #########################################################

        max_epochs = max_epochs
        val_interval = val_interval
        best_mse = np.inf   # init to infinity

        history_loss_validation = []
        history_loss_training = []
        loss_fn = nn.L1Loss().to(device) 
        New_PerceptualLoss = Accelerated_PerceptualLoss().to(device)
        Fake3D_PerceptualLoss = Accelerated_Psudo3D_PerceptualLoss().to(device)
        
        mae = MeanAbsoluteError().to(device)
        psnr = PeakSignalNoiseRatio().to(device)
        ssim = StructuralSimilarityIndexMeasure().to(device) 

        Combined_Model_Params = list(Nature_Model.parameters())
        optimizer = optim.AdamW(Combined_Model_Params, lr=learning_rate) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

        mask = torch.zeros((1,number_of_output_slices, 128, 128))
        start = 32  # Starting index for the region
        end = start + 64  # End index for the region
        mask[:,:, start:end, :] = 1
        mask = mask.to(device)


        ############# trasining and Validaiton Framework #############
        for epoch in range(max_epochs):
            print("---" * 20)
            print(f"==================== Epoch {epoch + 1}/{max_epochs} ====================")
            print("---" * 20)

            Nature_Model.train() 

            Training_loss =[]
            PSNR_scores_Training = [] 
            SSIM_scores_Training = []
            MAE_Training_masked = [] 
            PSNR_scores_Training_masked = [] 
            SSIM_scores_Training_masked = []
            MSE_only_loss_train = []

            for train_data in progressbar.progressbar(train_loader):

                CT_data = train_data["img"][:,:,:,:,:number_of_output_slices]
                labels = train_data["label"][:,:,:,:,:number_of_output_slices]
                PTV_struc = train_data["PTV"][:,:,:,:,:number_of_output_slices]
                Bladder_struct = train_data["Bladder"][:,:,:,:,:number_of_output_slices] 
                Rectum_struct  = train_data["Rectum"][:,:,:,:,:number_of_output_slices]
    
                CT_data = CT_data.squeeze(0).permute(0,-1,1,2)          #(1,1,128,128,64) --> #(1,64,128,128)
                labels_CBCT = labels.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
                PTV_struc = PTV_struc.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
                Bladder_struct = Bladder_struct.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
                Rectum_struct = Rectum_struct.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 

                input_to_Model_DRR = DRR_generator_CBCT_DRRs(labels_CBCT, angles=input_angles_DRRs)         #(1,projections,128,128)
                input_to_Model_DRR = torch.flipud(input_to_Model_DRR)
                
                input_to_Model_DRR = input_to_Model_DRR.to(device = device)
                CT_data_to_model = CT_data.unsqueeze(0).to(device = device) * mask         #(1,1,64,128,128)
                labels_CBCT = labels_CBCT.to(device) * mask 
                OAR_masks = Anatomy_Mask(PTV=PTV_struc, Bladder=Bladder_struct, Rectum=Rectum_struct)
                OAR_masks = OAR_masks.to(device)
                
                ############# Train the Model #############
                sCBCT = Nature_Model(input_to_Model_DRR, CT_data_to_model)
                sCBCT = sCBCT * mask
                sDRR = DRR_generator_CBCT_DRRs(sCBCT, angles=input_angles_DRRs)
                DRR = DRR_generator_CBCT_DRRs(labels_CBCT, angles=input_angles_DRRs)
                sCBCT = sCBCT.to(device)
                OAR_masked_sCBCT = sCBCT * OAR_masks
                OAR_masked_CBCT = labels_CBCT * OAR_masks
                ############# Loss & Metrics #############
                
                MSE_loss = loss_fn(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:])
                Percept_loss =  Fake3D_PerceptualLoss(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:])
                OAR_Percept_loss =  Fake3D_PerceptualLoss(OAR_masked_sCBCT[:,:,start:end,:], OAR_masked_CBCT[:,:,start:end,:])
                Percept_loss_DRR =  New_PerceptualLoss(sDRR, DRR)
                
                loss = alpha * MSE_loss + beta * Percept_loss + gamma * Percept_loss_DRR + omega * OAR_Percept_loss
                
                MAE_Training_masked.append(mae(sCBCT[:,:,start:end,:].contiguous(), target=labels_CBCT[:,:,start:end,:].contiguous()).item())
                SSIM_scores_Training_masked.append(ssim(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:]).item())
                PSNR_scores_Training_masked.append(psnr(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:]).item())
                SSIM_scores_Training.append(ssim(sCBCT[:,:,:,:], labels_CBCT[:,:,:,:]).item())
                PSNR_scores_Training.append(psnr(sCBCT[:,:,:,:], labels_CBCT[:,:,:,:]).item())

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                MSE_only_loss_train.append(MSE_loss.item())
                Training_loss.append(loss.item())
            

            wandb.log({
            "epoch": epoch + 1,
            "MSE_Train": np.mean(MSE_only_loss_train),
            "Loss_Training": np.mean(Training_loss),
            "SSIM_Training": np.mean(SSIM_scores_Training),
            "PSNR_Training": np.mean(PSNR_scores_Training),
            "MAE_Training_masked": np.mean(MAE_Training_masked),
            "SSIM_Training_masked": np.mean(SSIM_scores_Training_masked),
            "PSNR_Training_masked": np.mean(PSNR_scores_Training_masked),
            })
                

            history_loss_training.append(np.mean(Training_loss))
            slice_selector = np.random.randint(35, 45)

            CT_data = CT_data_to_model.squeeze(0)
            image_list= [
            torch.sin(CT_data[0,slice_selector,:,:].T)**2, 
            torch.flipud(torch.sin(input_to_Model_DRR[0,0,:,:])**2),
            torch.sin(OAR_masked_sCBCT[0,slice_selector,:,:].T)**2,
            torch.sin(OAR_masked_CBCT[0,slice_selector,:,:].T)**2,
            torch.sin(sCBCT[0,slice_selector,start:end,:].T)**2,
            torch.sin(labels_CBCT[0,slice_selector,start:end,:].T)**2,
            
            ]

            list_of_fig_titles = [ "CT","DRR","OARs_sCBCT","OARs_CBCT" ,"sCBCT", "CBCT"]

            plot_images_in_grid(image_list, list_of_fig_titles, os.path.join(save_images_path, Method_name,  changed_variables, f"Image_Train_ep{epoch+1}"))                
            print(f"==== Epoch {epoch + 1} Training Loss: {np.mean(Training_loss):.5f}, MAE: {np.round(np.mean(MAE_Training_masked),4)}, SSIM: {np.round(np.mean(SSIM_scores_Training_masked),4)}, PSNR: {np.round(np.mean(PSNR_scores_Training_masked),4)} ==== ====")
                        
            ############### The Validation Dataset ###############
            
            if (epoch + 1) % val_interval == 0:

                Nature_Model.train() 

                average_total_val_loss = []
                PSNR_scores_Validation = [] 
                SSIM_scores_Validation = []
                MAE_Validation_masked = [] 
                PSNR_scores_Validation_masked = [] 
                SSIM_scores_Validation_masked = []
                MSE_only_loss_val = []


                with torch.no_grad():
                    for val_data in progressbar.progressbar(val_loader):

                        CT_data = val_data["img"][:,:,:,:,:number_of_output_slices]
                        labels = val_data["label"][:,:,:,:,:number_of_output_slices]
                        PTV_struc = val_data["PTV"][:,:,:,:,:number_of_output_slices]
                        Bladder_struct = val_data["Bladder"][:,:,:,:,:number_of_output_slices] 
                        Rectum_struct  = val_data["Rectum"][:,:,:,:,:number_of_output_slices]
            
                        CT_data = CT_data.squeeze(0).permute(0,-1,1,2)          #(1,1,128,128,64) --> #(1,64,128,128)
                        labels_CBCT = labels.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
                        PTV_struc = PTV_struc.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
                        Bladder_struct = Bladder_struct.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
                        Rectum_struct = Rectum_struct.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 

                        input_to_Model_DRR = DRR_generator_CBCT_DRRs(labels_CBCT, angles=input_angles_DRRs)         #(1,projections,128,128)
                        input_to_Model_DRR = torch.flipud(input_to_Model_DRR)
                        
                        input_to_Model_DRR = input_to_Model_DRR.to(device = device)
                        CT_data_to_model = CT_data.unsqueeze(0).to(device = device) * mask        #(1,1,64,128,128)
                        labels_CBCT = labels_CBCT.to(device) * mask 
                        OAR_masks = Anatomy_Mask(PTV=PTV_struc, Bladder=Bladder_struct, Rectum=Rectum_struct)
                        OAR_masks = OAR_masks.to(device)
                        
                        
                        ############# Validate the Model #############
                        sCBCT = Nature_Model(input_to_Model_DRR, CT_data_to_model)
                        sCBCT = sCBCT * mask
                        sDRR = DRR_generator_CBCT_DRRs(sCBCT, angles=input_angles_DRRs)
                        DRR = DRR_generator_CBCT_DRRs(labels_CBCT, angles=input_angles_DRRs)
                        sCBCT = sCBCT.to(device)
                        OAR_masked_sCBCT = sCBCT * OAR_masks
                        OAR_masked_CBCT = labels_CBCT * OAR_masks                   
                        ############# Loss & Metrics #############

                        MSE_loss = loss_fn(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:])
                        Percept_loss =  Fake3D_PerceptualLoss(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:])
                        OAR_Percept_loss =  Fake3D_PerceptualLoss(OAR_masked_sCBCT[:,:,start:end,:], OAR_masked_CBCT[:,:,start:end,:])
                        Percept_loss_DRR =  New_PerceptualLoss(sDRR, DRR)
                        
                        loss_validation = alpha * MSE_loss + beta * Percept_loss + gamma * Percept_loss_DRR + omega * OAR_Percept_loss
                        
                        MAE_Validation_masked.append(mae(sCBCT[:,:,start:end,:].contiguous(), target=labels_CBCT[:,:,start:end,:].contiguous()).item())                        
                        SSIM_scores_Validation_masked.append(ssim(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:]).item())
                        PSNR_scores_Validation_masked.append(psnr(sCBCT[:,:,start:end,:], labels_CBCT[:,:,start:end,:]).item())
                        SSIM_scores_Validation.append(ssim(sCBCT[:,:,:,:], labels_CBCT[:,:,:,:]).item())
                        PSNR_scores_Validation.append(psnr(sCBCT[:,:,:,:], labels_CBCT[:,:,:,:]).item())
                        
                        MSE_only_loss_val.append(MSE_loss.item())
                        average_total_val_loss.append(loss_validation.item())

                    wandb.log({
                    "epoch": epoch + 1,
                    "MSE_Val": np.mean(MSE_only_loss_val),
                    "Loss_Validation": np.mean(average_total_val_loss),
                    "SSIM_Validation": np.mean(SSIM_scores_Validation),
                    "PSNR_Validation": np.mean(PSNR_scores_Validation),
                    "MAE_Validation_masked": np.mean(MAE_Validation_masked),
                    "SSIM_Validation_masked": np.mean(SSIM_scores_Validation_masked),
                    "PSNR_Validation_masked": np.mean(PSNR_scores_Validation_masked),
                    })

                    scheduler.step(np.mean(average_total_val_loss))
                    if np.mean(average_total_val_loss) < best_mse: 
                        best_mse = np.round(np.mean(average_total_val_loss),5)

                        torch.save({'epoch': epoch, 'model_state_dict': Nature_Model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': np.mean(average_total_val_loss)}, os.path.join(f"{Saving_path}",f"{Method_name}_{changed_variables}"+ ".pth"))
                        
                        wandb.save(f"{Saved_model_name}_{changed_variables}_epoch_{epoch+1}.pth")
                        #print(f"Almost Saved Model's best loss {best_mse}")    

                history_loss_validation.append(np.mean(average_total_val_loss))
                print(f"==== Epoch {epoch + 1} Validation Loss: {np.mean(average_total_val_loss):.5f}, MAE: {np.round(np.mean(MAE_Validation_masked),4)}, SSIM: {np.round(np.mean(SSIM_scores_Validation_masked),4)}, PSNR: {np.round(np.mean(PSNR_scores_Validation_masked),4)} ====")
            

                CT_data = CT_data_to_model.squeeze(0)
                image_list= [
                torch.sin(CT_data[0,slice_selector,:,:].T)**2, 
                torch.flipud(torch.sin(input_to_Model_DRR[0,0,:,:])**2),
                torch.sin(OAR_masked_sCBCT[0,slice_selector,:,:].T)**2,
                torch.sin(OAR_masked_CBCT[0,slice_selector,:,:].T)**2,
                torch.sin(sCBCT[0,slice_selector,start:end,:].T)**2,
                torch.sin(labels_CBCT[0,slice_selector,start:end,:].T)**2,
                
                ]

                plot_images_in_grid(image_list, list_of_fig_titles,  os.path.join(save_images_path, Method_name, changed_variables, f"Image_Val_ep{epoch+1}"))  

        wandb.finish()

        time.sleep(30)



if __name__ == '__main__':
    train()
