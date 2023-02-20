import torchio as tio
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu
from pyrobex.robex import robex

class Preprocessing:

    def __init__(self, input_folder, output_folder):
        """
        @input_folder: the folder that has the scans you want to preprocess
        @output_folder: the folder that the preprocessed scans will be stored in 
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.files = os.listdir(input_folder)

    def correctBias(self, input_image_path):
        input_image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
        self.createMRIMask(input_image_path)
        maskImage = sitk.ReadImage("temp_mask.nii", sitk.sitkUInt8)
        os.remove("temp_mask.nii")
        shrinkFactor = 3

        image = sitk.Shrink(
            input_image, [shrinkFactor] * input_image.GetDimension()
        )
        maskImage = sitk.Shrink(
            maskImage, [shrinkFactor] * input_image.GetDimension()
        )

        numFittingLevels = 4

        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        corrected_image = corrector.Execute(image, maskImage)

        log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)

        corrected_image_full_resolution = input_image / \
            sitk.Exp(log_bias_field)

        sitk.WriteImage(corrected_image_full_resolution, "bixed.nii")

    def createMRIMask(self, input_image_path):
        img = nib.load(input_image_path)
        data = np.squeeze(img.get_fdata())
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
        nib.save(mask_img, "temp_mask.nii")

    def extractBrain(self):
        image = nib.load("./bixed.nii")
        stripped, mask = robex(image)
        sitk.Show(stripped)

    def removeExtension(self, input_path):
        input_path.replace(".nii", "")
        return input_path