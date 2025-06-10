from Collection_of_models import*
from Utilities import* 

def Infer():    
    import torch
    import nibabel as nib
    from Config_PreProc_Train_Param import CONFIG_Model_Training
    cuda_gpu = CONFIG_Model_Training["CUDA"]
    #device = torch.device(cuda_gpu if torch.cuda.is_available() else "cpu")
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
    from torchmetrics import MeanAbsoluteError
    from monai.utils import set_determinism
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import progressbar
    import numpy as np
    import time
    import wandb
    import glob
    import os  
    set_determinism(seed=0)
    Inference_loop = 3
    list_of_devices = ["cuda:1", "cuda:1", "cuda:1", "cuda:0"]
    device = list_of_devices[Inference_loop]
    print("Using Device:", device)

    #---------------------------------------------------------------------------------- Extended Testing --------------------------------------------------------------------------------
    patient_folder_path_Extended_test =  r"/home/mluser1/Musti_2D_3D/Crypted_Extended_Testset/Proc_w_Hero/"

    Img_CT_test_Extened = glob.glob(os.path.join(patient_folder_path_Extended_test,"**",'IMGS_*',"*.nii"))
    Label_CBCT_test_Extened = glob.glob(os.path.join(patient_folder_path_Extended_test,"**",'Labels_*',"*.nii")) #634 samples in Extended test set
    Data_files_test_Extened = patient_organizer(first_list_of_patient_imges=Img_CT_test_Extened, second_list_of_patient_images=Label_CBCT_test_Extened)

    Data_files_test_Extened = OAR_path_mapper_ExtenedDataset_test(path=patient_folder_path_Extended_test, list_of_patients_for_testing=Data_files_test_Extened) 
    
    testing_files = Data_files_test_Extened[:]

    print("Number of Patients Samples:", len(testing_files))
    List_of_patient_names = [(testing_files[pat]["img"].split("/")[-3]) for pat in range(len(testing_files))]    
    #---------------------------------------------------------------------------------- Data Loading --------------------------------------------------------------------------------
    from monai.data import DataLoader, CacheDataset
    from monai.transforms import (
        LoadImaged,
        Compose,
        EnsureTyped,
        Resized,
        CenterSpatialCropd,
        ScaleIntensityRanged
        )

    input_image_size = CONFIG_Model_Training["input_image_size"]
    batch_size = CONFIG_Model_Training["batch_size"]


    test_transforms = Compose(
        [
            LoadImaged(keys=["img","label", "PTV","Rectum","Bladder", "BODY"], ensure_channel_first=True),
            EnsureTyped(keys=["img","label", "PTV","Rectum","Bladder", "BODY"]),
            CenterSpatialCropd(keys=["img","label"], roi_size=(input_image_size,input_image_size,70)),
            Resized(keys=["PTV","Rectum","Bladder", "BODY"], mode="nearest", spatial_size=(256,256,88)),
            CenterSpatialCropd(keys=["PTV","Rectum","Bladder", "BODY"], roi_size=(input_image_size,input_image_size,70)),
            ScaleIntensityRanged(keys=["img"], a_min=-1024, a_max=2000, b_min=0, b_max=1, clip=True),
        ]
    )
    

    #-----------------------------------------------------------------------------------------------------
    #------------------------------------------- Start Experiment Testing -------------------------------
    #-----------------------------------------------------------------------------------------------------


    test_ds = CacheDataset( data=testing_files, transform=test_transforms, cache_rate=1.0, runtime_cache="processes", copy_cache=False)
    test_loader = DataLoader(test_ds, 
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=32,  
                            pin_memory=torch.cuda.is_available())

    number_of_output_slices = CONFIG_Model_Training["number_of_output_slices"]
    input_angles_DRRs =  CONFIG_Model_Training["input_angles_DRRs"]

    number_of_projections = len(input_angles_DRRs)
    number_of_output_slices = CONFIG_Model_Training["number_of_output_slices"]

    
    mask = torch.zeros((1,number_of_output_slices, 128, 128))
    start = 32  # Starting index for the region
    end = start + 64  # End index for the region
    mask[:,:, start:end, :] = 1
    mask = mask.to(device)

    
    
    #DL_Model = Model_4_DRRSOnly(in_channels_2d=number_of_projections, out_slices=number_of_output_slices).to(device)
    #DL_Model_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_4_OnlyDRRs_OAR_Light_DRR_3DLoss_beta_0.05_gamma_0.01_omega0.04.pth"

    #DL_Model = Model_4_Base_Simplified_Light(in_channels_2d=number_of_projections, in_channels_3d=1, out_slices=number_of_output_slices).to(device)
    #DL_Model_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_4_OAR_2skip_Light_DRR_3DLoss_beta_0.05_gamma_0.01_omega0.04.pth"           #CUDA1
    
    #DL_Model = Model_5_Base_2Skips_Simplified_Light(in_channels_2d=number_of_projections, in_channels_3d=1, out_slices=number_of_output_slices).to(device)
    #DL_Model_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_5_OAR_2skip_Light_DRR_3DLoss_beta_0.05_gamma_0.01_omega0.04.pth"           #CUDA0
    list_of_model_weight_paths = [r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_2skip_Light_DRR_3DLoss_beta_0.05_gamma_0.01_omega0.04.pth",
                                    r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_MAEandPL_Light_DRR_3DLoss_beta_0.05_gamma_0.0_omega0.0.pth",
                                    r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_LossEditedNoMAE_Light_DRR_3DLoss_beta_0.05_gamma_0.0_omega0.0.pth",
                                    r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_LossEdited_Light_DRR_3DLoss_beta_0.0_gamma_0.0_omega0.0.pth"]                                      
  

    DL_Model = Model_6_Base_2Skip_EasyRes_Simplified_Light(in_channels_2d=number_of_projections, in_channels_3d=1, out_slices=number_of_output_slices).to(device)
    #DL_Model_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_2skip_Light_DRR_3DLoss_beta_0.05_gamma_0.01_omega0.04.pth"           #CUDA1 ALF
    #DL_Model_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_MAEandPL_Light_DRR_3DLoss_beta_0.05_gamma_0.0_omega0.0.pth"          #CUDA1 
    #DL_Model_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_LossEditedNoMAE_Light_DRR_3DLoss_beta_0.05_gamma_0.0_omega0.0.pth"   #CUDA1
    #DL_Model_path = r"/home/mluser1/Musti_2D_3D/Some_Results/ModelsWeights/MaskedOAR_Model_6_OAR_LossEdited_Light_DRR_3DLoss_beta_0.0_gamma_0.0_omega0.0.pth"          #CUDA0
    Model = DL_Model
    checkpoint = torch.load(list_of_model_weight_paths[Inference_loop])
    Model.load_state_dict(checkpoint['model_state_dict'])
    Model.eval()  # Set the model to evaluation mode

    folder_name = "Extended_Inference_To_Article_" + list_of_model_weight_paths[Inference_loop].split("/")[-1]  #Method_N_Inference
    output_saving_path = r"/home/mluser1/Musti_2D_3D/Inference_Output/"
    output_saving_path = os.path.join(output_saving_path, folder_name)
    os.makedirs(output_saving_path, exist_ok=True)


    MAE_test_masked =[] 
    LPIP_scores_test_masked =[] 
    SSIM_scores_test_masked =[] 
    PSNR_scores_test_masked =[] 
    patient_scan_counter = 0
    patient_inference_times = []

    LPIP = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device) 
    
    list_of_loss_names = ["FusionSkipRes_ALF_Masked", "FusionSkipRes_MAE&PL_Masked", "FusionSkipRes_onlyPL_Masked", "FusionSkipRes_onlyMAE_Masked"]
    wandb.init(project="Extended_Output_Masked", name= list_of_loss_names[Inference_loop] ,config={"number_of_patient_samples": len(testing_files)})

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        ############# 10 GPU warmup rounds ###########
        for _ in range(10):
            dummy_input_DRR = torch.randn(1, number_of_projections, 128, 128).to(device)
            dummy_input_CT = torch.randn(1, 1, number_of_output_slices, 128, 128).to(device)
            _ = Model(dummy_input_DRR, dummy_input_CT)
        ############# Inference Loop #############
        for test_data in progressbar.progressbar(test_loader):

            CT_data = test_data["img"][:,:,:,:,:number_of_output_slices]
            labels = test_data["label"][:,:,:,:,:number_of_output_slices]
            PTV_struc = test_data["PTV"][:,:,:,:,:number_of_output_slices]
            Bladder_struct = test_data["Bladder"][:,:,:,:,:number_of_output_slices] 
            Rectum_struct  = test_data["Rectum"][:,:,:,:,:number_of_output_slices]
            BODY_mask = test_data["BODY"][:,:,:,:,:number_of_output_slices]
            
            CT_data = CT_data.squeeze(0).permute(0,-1,1,2)          #(1,1,128,128,64) --> #(1,64,128,128)
            labels_CBCT = labels.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
            PTV_struc = PTV_struc.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
            Bladder_struct = Bladder_struct.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
            Rectum_struct = Rectum_struct.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 
            BODY_mask = BODY_mask.squeeze(0).permute(0,-1,1,2)       #(1,1,128,128,64) --> #(1,64,128,128) 

            labels_CBCT_clamped_HU = torch.clamp(labels_CBCT, min=-1024, max=2000)
            labels_CBCT_normalized, _, _ = HU_to_Norm(labels_CBCT_clamped_HU)
            labels_CBCT_Normed_cropped = labels_CBCT_normalized.to(device) * mask #[0,1] 
            
            input_to_Model_DRR = DRR_generator_CBCT_DRRs(labels_CBCT_normalized, angles=input_angles_DRRs)         #(1,projections,128,128)
            input_to_Model_DRR = torch.flipud(input_to_Model_DRR)
            
            input_to_Model_DRR = input_to_Model_DRR.to(device = device)
            CT_data_to_model = CT_data.unsqueeze(0).to(device = device) * mask          #(1,1,64,128,128)
            BODY_mask = BODY_mask.to(device) * mask 
            

            ############# Start Infer With The Model #############
            starter.record()
            sCBCT = Model(input_to_Model_DRR, CT_data_to_model)
            ender.record()
            #sCBCT = Model(input_to_Model_DRR)
            sCBCT = sCBCT * mask #(1,64,128,128)
            torch.cuda.synchronize()  # Waits for everything to finish running
            time_end = starter.elapsed_time(ender) #/ 1000  # if we need to convert to seconds
            patient_inference_times.append(time_end)
            ############# End Infer With The Model #############
            labels_CBCT_clamped_HU = labels_CBCT_clamped_HU.to(device) * mask
            sCBCT_De_Norm_HU = Norm_to_HU(norm_image=sCBCT, im_min=-1024, im_max=2000) * mask

            BODY_mask = BODY_mask.detach().cpu().numpy()[0]                        #(1,64,128,128)
            sCBCT = torch.clamp(sCBCT, min=0, max=1)
            sCBCT_npy_norm = sCBCT.detach().cpu().numpy()[0]
            sCBCT_De_Norm_HU = sCBCT_De_Norm_HU.detach().cpu().numpy()[0]
            CBCT_np_HU = labels_CBCT_clamped_HU.detach().cpu().numpy()[0]
            CBCT_np_Norm = labels_CBCT_Normed_cropped.detach().cpu().numpy()[0]


            mae_value =  Masked_MAE(predicted= sCBCT_De_Norm_HU,  target=CBCT_np_HU, mask=BODY_mask)
            SSIM_value = Masked_SSIM(predicted=sCBCT_npy_norm, target=CBCT_np_Norm, mask=BODY_mask, data_range=1)
            PSNR_value = Masked_PSNR(predicted=sCBCT_npy_norm, target=CBCT_np_Norm, mask=BODY_mask, data_range=1)

            
            LPIP_sCBCT = sCBCT.squeeze(0).unsqueeze(1).repeat(1, 3, 1, 1)
            LPIP_CBCT = labels_CBCT_Normed_cropped.squeeze(0).unsqueeze(1).repeat(1, 3, 1, 1) 
            LPIP_scores_test_masked.append(LPIP(LPIP_sCBCT[:,:,start:end,:], LPIP_CBCT[:,:,start:end,:]).item()) 
            
            MAE_test_masked.append(mae_value)
            SSIM_scores_test_masked.append(SSIM_value)
            PSNR_scores_test_masked.append(PSNR_value)
            
            patient_name = List_of_patient_names[patient_scan_counter] 
            os.makedirs(os.path.join(output_saving_path,f"pat_{patient_name}_{patient_scan_counter}" ), exist_ok=True)

            patient_MAE_masked = np.round(MAE_test_masked[patient_scan_counter], 3)
            patient_SSIM_score_masked = np.round(SSIM_scores_test_masked[patient_scan_counter], 3)
            patient_PSNR_score_masked = np.round(PSNR_scores_test_masked[patient_scan_counter], 3)
            patient_LPIP_score_masked = np.round(LPIP_scores_test_masked[patient_scan_counter], 3)

            #if patient_PSNR_score_masked > 31:
            #    print("Patient Name:", patient_name + "_" + str(patient_scan_counter))
            #    print("MAE:", patient_MAE_masked)
            #    print("SSIM:", patient_SSIM_score_masked)
            #    print("PSNR:", patient_PSNR_score_masked)
            #    print("LPIP:", patient_LPIP_score_masked)
            
            wandb.log({
            "Patient_scan_ID": patient_scan_counter,
            "patient_MAE_masked": patient_MAE_masked,
            "patient_SSIM_masked": patient_SSIM_score_masked,
            "patient_PSNR_masked": patient_PSNR_score_masked,
            "patient_LPIP_masked": patient_LPIP_score_masked,
            })
#
            PTV = PTV_struc.detach().cpu().numpy()
            Bladder = Bladder_struct.detach().cpu().numpy()
            Rectum = Rectum_struct.detach().cpu().numpy()
            CT_data_to_model = CT_data_to_model.detach().cpu().numpy()
            input_to_Model_DRR = input_to_Model_DRR.detach().cpu().numpy()
            affine = np.eye(4)
            PTV = PTV[0]
            Bladder = Bladder[0]
            Rectum = Rectum[0]
            CT_data_npy = CT_data_to_model[0][0] 
            input_to_model_npy = input_to_Model_DRR[0]
            sCBCT_nifti_img = nib.Nifti1Image(sCBCT_De_Norm_HU, affine)
            PTV_nifti_img = nib.Nifti1Image(PTV, affine)
            Bladder_nifti_img = nib.Nifti1Image(Bladder, affine)
            Rectum_nifti_img = nib.Nifti1Image(Rectum, affine)
            BODY_nifti_img = nib.Nifti1Image(BODY_mask, affine)
            CT_data_nifti_img = nib.Nifti1Image(CT_data_npy, affine)
            labels_CBCT_nifti_img = nib.Nifti1Image(CBCT_np_HU, affine)
            input_to_model_img = nib.Nifti1Image(input_to_model_npy, affine)
            nib.save(sCBCT_nifti_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + f"sCBCT" + ".nii.gz")
            nib.save(PTV_nifti_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + "PTV_" + str(patient_scan_counter) + ".nii.gz")
            nib.save(Bladder_nifti_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + "Bladder_" + str(patient_scan_counter) + ".nii.gz")
            nib.save(Rectum_nifti_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + "Rectum_" + str(patient_scan_counter) + ".nii.gz")
            nib.save(BODY_nifti_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + "BODY_" + str(patient_scan_counter) + ".nii.gz")
            nib.save(CT_data_nifti_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + "CT" + str(patient_scan_counter) + ".nii.gz")
            nib.save(labels_CBCT_nifti_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + "CBCT_" + str(patient_scan_counter) + ".nii.gz")
            nib.save(input_to_model_img, output_saving_path + "/" + f"pat_{patient_name}_{patient_scan_counter}" + "/" + "DRR_" + str(patient_scan_counter) + ".nii.gz")
            
            patient_scan_counter += 1

    print("-------------------------------------")
    print("-------------------------------------")
    
    wandb.log({
    "Mean MAE_Test_masked": np.round(np.mean(MAE_test_masked),3),
    "Mean SSIM_Test_masked": np.round(np.mean(SSIM_scores_test_masked),3),
    "Mean LPIP_Test_masked": np.round(np.mean(LPIP_scores_test_masked),3),
    "Mean PSNR_Test_masked": np.round(np.mean(PSNR_scores_test_masked),3),
    "Mean Inference Time": np.round(np.mean(patient_inference_times),3),
    })
    
    
    #print("Mean MAE_Test_masked", np.round(np.mean(MAE_test_masked),3))
    #print("Mean SSIM_Test_masked", np.round(np.mean(SSIM_scores_test_masked),3))
    #print("Mean LPIP_Test_masked", np.round(np.mean(LPIP_scores_test_masked),3))
    #print("Mean PSNR_Test_masked", np.round(np.mean(PSNR_scores_test_masked),3))
    #print("Mean Inference Time", np.round(np.mean(patient_inference_times),3))
    #print("Inference Done!")

    print("-------------------------------------")
    print("-------------------------------------")




if __name__ == '__main__':
    Infer()
