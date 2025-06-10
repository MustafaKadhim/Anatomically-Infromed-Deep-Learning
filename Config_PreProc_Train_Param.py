import numpy as np

CONFIG_PreProc = {
    "Patient_group_name": "70Gy",
}


CONFIG_Model_Training = {
    "number_of_output_slices" : 64,
    "n_feat" : 128,
    "input_image_size" : 128,
    "input_angles_DRRs":  [0, 3.1415/2],
    "source_to_detec": 10,

    "batch_size": 1,
    "val_interval": 1,
    "max_epochs": 40,
    "learning_rate": 1e-3,
    "alpha": 1,    #[1, 0] 
    "beta":  0.05,  #[1, 0.5, 0.1, 0.05, 0.01, 0]  Good = 0.05
    "gamma":  0.01,  #[1, 0.5, 0.1, 0.05, 0.01, 0] Good = 0.01
    "omega":  0.04,  #[1, 0.5, 0.1, 0.05, 0.01, 0] Good = 0.04
    "CUDA": "cuda:1"
}