from nipype.interfaces import fsl
from pyrobex.robex import robex
import nibabel as nib
import numpy as np
import os
from dipy.segment.mask import median_otsu
from tqdm import tqdm
import SimpleITK as sitk
import torch

def createMRIMask(input_image_path):
    img = nib.load(input_image_path)
    data = np.squeeze(img.get_fdata())
    b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
    mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
    nib.save(mask_img, "temp_mask.nii")

def correctBias(input_image_path):
    input_image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
    createMRIMask(input_image_path)    
    mask_image = sitk.ReadImage("temp_mask.nii", sitk.sitkUInt8)
    os.remove("temp_mask.nii")
    shrinkFactor = 3
    shrunk_image = sitk.Shrink(
        input_image, [shrinkFactor] * input_image.GetDimension()
    )
    shrunk_mask = sitk.Shrink(
        mask_image, [shrinkFactor] * input_image.GetDimension()
    )
    numFittingLevels = 4
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(shrunk_image, shrunk_mask)
    log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
    corrected_image_full_resolution = input_image / \
        sitk.Exp(log_bias_field)
    sitk.WriteImage(corrected_image_full_resolution, input_image_path)

def minMaxNormalization(image):
    tensor_min = torch.min(image)
    tensor_max = torch.max(image)
    normalized_image = (image - tensor_min) / (tensor_max - tensor_min)
    return normalized_image

def correct(input_image, input_directory, output_directory):
    image_name = input_image[:-4]

    mcflt = fsl.MCFLIRT(in_file=f"{input_directory}/{image_name}.nii")
    res = mcflt.run()

    nii_image = nib.load(f"{image_name}_mcf.nii.gz")
    nii_data = nii_image.get_fdata()
    extracted_nii = nib.Nifti1Image(nii_data, nii_image.affine)
    nib.save(extracted_nii, f"{output_directory}/{image_name}_mcf.nii")

    n4_input = f"{output_directory}/{image_name}_mcf.nii"
    correctBias(n4_input)

    image = nib.load(n4_input)
    skull_stripped_image, mask = robex(image=image)
    nii_data = skull_stripped_image.get_fdata()
    tensor_data = torch.tensor(nii_data, dtype=torch.float32)
    normalized_tensor = minMaxNormalization(tensor_data) 

    normalized_array = normalized_tensor.numpy()
    normalized_image = nib.Nifti1Image(normalized_array, affine=np.eye(4))
    nib.save(normalized_image, f"./{output_directory}/{image_name}.nii")

    os.remove(f"{image_name}_mcf.nii.gz")
    os.remove(f"{output_directory}/{image_name}_mcf.nii")

for i in tqdm(os.listdir("./data/raw_input")):
    correct(i, "./data/raw_input", "./data/processed_input")