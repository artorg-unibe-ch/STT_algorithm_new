# imports

from Pipeline_Modules.clinical_CT_soft_tissue_extraction import *
from Pipeline_Modules.stt import *

start_time = time.time()

# --- Define directories ---
# INS folder directories
ct_folder_affirmct_INS_uncal = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/INS_cropped')
ct_folder_affirmct_INS_cal = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/INS_cropped_calib/INS_in-scan_calib')
seg_folder_affirmct_INS = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/INS_seg_femur')
calibration_file_INS = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/metadata_INS.csv')

# HUG folder directories
ct_folder_affirmct_HUG_uncal = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/HUG_cropped')
ct_folder_affirmct_HUG_cal = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/HUG_cropped_calib/HUG_asyn_calib_LP')
seg_folder_affirmct_HUG = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/HUG_seg_femur')
calibration_file_HUG = str('/Volumes/artorg_msb/projects/2020-AFFIRMCT/DATA/CT/AFFIRM-CT_14_07_23/metadata_HUG.csv')

# --- Run script ---
# Insel
run_stt(ct_folder_affirmct_INS_uncal, seg_folder_affirmct_INS, 'affirm_ct',
        'output_affirmct_INS_cal_newthreshold1.csv', 'uncalibrated',
        filename_input=None,
        calibration_file=calibration_file_INS)
# Geneva
# run_stt(ct_folder_affirmct_HUG_uncal, seg_folder_affirmct_HUG, 'affirm_ct',
#         'output_affirmct_HUG_cal_newthreshold3.csv', 'calibrated',
#         filename_input=None, calibration_file=calibration_file_HUG)

print()
print(f"Time elapsed: {(time.time() - start_time):.2f} seconds")
