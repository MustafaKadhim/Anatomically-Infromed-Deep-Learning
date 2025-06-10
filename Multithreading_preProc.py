from concurrent.futures import ThreadPoolExecutor
import os
import glob
import shutil
import time
from pydicom import dcmread

def process_patient(patient, main_path, output_dir):
    try:
        # Process plans in the patient folder
        plans_in_patient = glob.glob(os.path.join(main_path, patient, 'RP.*', "*.dcm"))
        for plan in plans_in_patient:
            ds_plan = dcmread(plan, specific_tags=[[0x300C, 0x002], [0x0020, 0x000D], [0x0008, 0x114A],[0x0008, 0x1155], [0x0008, 0x0070], [0x0008, 0x0060]], stop_before_pixels=True)
            plan_name = plan.split("\\")[-2]  # PlanName
            plan_SOP_instance_UID = ds_plan.StudyInstanceUID

            # Process CT/CBCT scans
            for CT_CBCT_scans in glob.glob(os.path.join(main_path, patient, 'C*', "*.dcm")):
                ds_ct_cbct = dcmread(CT_CBCT_scans, specific_tags=[[0x300C, 0x002], [0x0020, 0x000D], [0x0008, 0x114A], [0x0008, 0x1155], [0x0008, 0x0070], [0x0008, 0x0060]], stop_before_pixels=True)
                if ds_ct_cbct.Manufacturer == "Varian Medical Systems":
                    Referenced_SOP_Instance_UID = ds_ct_cbct.StudyInstanceUID

                    if plan_SOP_instance_UID == Referenced_SOP_Instance_UID:
                        os.makedirs(os.path.join(output_dir, patient, plan_name, CT_CBCT_scans.split("\\")[-2]), exist_ok=True)
                        shutil.copy2(CT_CBCT_scans, os.path.join(output_dir, patient, plan_name, CT_CBCT_scans.split("\\")[-2]))
                else:
                    os.makedirs(os.path.join(output_dir, patient, CT_CBCT_scans.split("\\")[-2]), exist_ok=True)
                    shutil.copy2(CT_CBCT_scans, os.path.join(output_dir, patient, CT_CBCT_scans.split("\\")[-2]))

            # Copy plan files
            shutil.copy2(plan, os.path.join(output_dir, patient, plan_name))

        # Process RS files
        RS_file = glob.glob(os.path.join(main_path, patient, 'RS*', "*.dcm"))
        if RS_file:
            RS_file = RS_file[0]
            os.makedirs(os.path.join(output_dir, patient, RS_file.split("\\")[-2]), exist_ok=True)
            shutil.copy2(RS_file, os.path.join(output_dir, patient, RS_file.split("\\")[-2]))
    except Exception as e:
        print("Trouble with:", patient, "Error:", str(e))

# Main execution
if __name__ == "__main__":

    main_path = r"F:\2D_3D_Project\Preproc_Exrtatest_patients"
    output_dir = r"F:\2D_3D_Project\Preproc_Exrtatest_patients\Ordering_Preproc_Exrtatest_patients"
    patients = os.listdir(main_path)
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        list(executor.map(process_patient, patients, [main_path]*len(patients), [output_dir]*len(patients)))


    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")