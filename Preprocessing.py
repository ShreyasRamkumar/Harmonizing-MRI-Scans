import torchio as tio
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu
from pyrobex import robex
from tqdm import tqdm
import torch

class Preprocessing:

    def __init__(self, input_folder, output_folder):
        """
        @input_folder: the folder that has the scans you want to preprocess
        @output_folder: the folder that the preprocessed scans will be stored in 
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        

    def run_preprocess(self):
        for i in tqdm(self.files):
            self.correctBias(input_image_path=f"./{self.input_folder}/{i}")
            self.extractBrain(i)
            print("successful!\n")
        print("Preprocessing Complete!")

    def correctBias(self, input_image_path=None, image_created = False, image = None, image_mask = None):

        input_image = image

        if image_created == False:
            input_image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
            self.createMRIMask(input_image_path)
        
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

        if image_created == False:
            sitk.WriteImage(corrected_image_full_resolution, "bixed.nii")
        
        elif image_created == True:
            return corrected_image_full_resolution
            print("Bias corrected!")

    def createMRIMask(self, input_image_path, image_created = False):
        img = nib.load(input_image_path)
        data = np.squeeze(img.get_fdata())
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
        nib.save(mask_img, "temp_mask.nii")
        print("Mask created!")

    def extractBrain(self, input_image_path):
        image = nib.load("./bixed.nii")
        stripped = robex.robex_stripped(image)
        output_path = input_image_path[:-4]
        nib.save(stripped, f"./{self.output_folder}/{output_path}_preprocessed.nii")
        os.remove("./bixed.nii")

    def minMaxNormalization(self, image):
        tensor_min = torch.min(image)
        tensor_max = torch.max(image)
        normalized_image = (image - tensor_min) / (tensor_max - tensor_min)
        return normalized_image
    
    def baselineImprov(self):
        denoise = sitk.MinMaxCurvatureFlowImageFilter()
        for i in tqdm(os.listdir(self.input_folder)):
            image_path = os.listdir(f"{self.input_folder}/{i}/anat")
            image = sitk.ReadImage(f"{self.input_folder}/{i}/anat/{image_path}", sitk.sitkFloat32)
            image_mask = self.createMRIMask(image_path, image_created=True)
            bias_corrected_image = self.correctBias(input_image_path=None, image_created=True, image=image, image_mask=image_mask)
            denoised_image = denoise.Execute(bias_corrected_image)
            print("Image denoised!")
            image_array = sitk.GetArrayFromImage(denoised_image)
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
            normalized_image = self.minMaxNormalization(image_tensor)
            print("Image normalized!")
            norm_img_array = np.array(normalized_image)
            norm_img_nifti = nib.Nifti1Image(norm_img_array, affine=np.eye(4))
            nib.save(norm_img_nifti, f"{self.output_folder}/{i}")
            print("Enhancement Successful!")

class MRITransform:
    def __init__(self, input_folder_path, output_folder_path) -> None:
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.input_files = os.listdir(input_folder_path)
    
    def create_training_data(self):
        transforms = tio.transforms.Compose([tio.transforms.RandomBiasField(coefficients=1, order=3), 
                                             tio.transforms.RandomMotion(degrees=4, translation=4, num_transforms=2), 
                                             tio.transforms.RandomNoise(mean=2, std=1)])
        for i in tqdm(self.input_files):
            image = tio.ScalarImage(f"{self.input_folder_path}/{i}")
            augmented_image = transforms(image)
            augmented_image.save(f"{self.output_folder_path}/{i}")


