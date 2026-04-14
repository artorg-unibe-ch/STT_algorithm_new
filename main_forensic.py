# imports

from Pipeline_Modules.clinical_CT_soft_tissue_extraction import *
from Pipeline_Modules.stt import *

start_time = time.time()

# FORENSIC
ct_folder_forensic_uncal = str('/Volumes/artorg_msb/projects/2015-FORENSIC/2_Forensic_donors/Cropped_uncalibrated')
ct_folder_forensic_cal = str('/Volumes/artorg_msb/projects/2015-FORENSIC/2_Forensic_donors/Calibrations/3_Calibrated_hip/1_Hard_kernel')
seg_folder_forensic = str('/Volumes/artorg_msb/projects/2015-FORENSIC/2_Forensic_donors/Calibrations/3b_Segmentation_hip/Segmentation_femur_nnunet')




run_stt(ct_folder_forensic_uncal, seg_folder_forensic, 'forensic', 'forensic_test.csv', filename_input=None, calibration_status='uncalibrated')


print()
print(f"Time elapsed: {(time.time() - start_time):.2f} seconds")