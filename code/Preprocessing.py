import torchio as tio
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu
from pyrobex import robex
from tqdm import tqdm
import torch
import nipype

# test for github push

class Preprocessing:
    """A class for preprocessing MRI scans.

    Attributes:
        input_folder (str): The folder containing the scans to preprocess.
        output_folder (str): The folder where preprocessed scans will be stored.
    """
    def __init__(self, input_folder, output_folder):
        """
        Initializes the Preprocessing class with input and output folders.

        Args:
            input_folder (str): The folder containing the scans to preprocess.
            output_folder (str): The folder where preprocessed scans will be stored.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        

    def run_preprocess(self):
        """Runs the preprocessing steps for all scans in the input folder."""
        for i in tqdm(self.files):
            self.correctBias(input_image_path=f"./{self.input_folder}/{i}")
            self.extractBrain(i)
            print("successful!\n")
        print("Preprocessing Complete!")

    def correctBias(self, input_image_path=None, image_created = False, image = None, image_mask = None):
        """
        Corrects bias in the MRI scan.

        Args:
            input_image_path (str): Path to the input MRI image.
            image_created (bool): Indicates if the image has already been created.
            image: The image object.
            image_mask: The mask image object.

        Returns:
            sitk.Image: The bias-corrected MRI image.
        """
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
        """
        Creates a mask for the MRI scan.

        Args:
            input_image_path (str): Path to the input MRI image.
            image_created (bool): Indicates if the image has already been created.

        Returns:
            None
        """
        img = nib.load(input_image_path)
        data = np.squeeze(img.get_fdata())
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
        nib.save(mask_img, "temp_mask.nii")
        print("Mask created!")

    def extractBrain(self, input_image_path):
        """
        Performms brain extraction on the MRI scan.

        Args:
            input_image_path (str): Path to the input MRI image.

        Returns:
            None
        """
        image = nib.load("./bixed.nii")
        stripped = robex.robex_stripped(image)
        output_path = input_image_path[:-4]
        nib.save(stripped, f"./{self.output_folder}/{output_path}_preprocessed.nii")
        os.remove("./bixed.nii")

    def minMaxNormalization(self, image):
        """
        Performs min-max normalization on the image.

        Args:
            image: The input image.

        Returns:
            torch.Tensor: The normalized image.
        """
        tensor_min = torch.min(image)
        tensor_max = torch.max(image)
        normalized_image = (image - tensor_min) / (tensor_max - tensor_min)
        return normalized_image

class DatasetPreparation(Preprocessing):
    """
    Prepares training and testing datasets using methods from Preprocessing class

    Attributes:
        input_folder_path (str): The folder containing the scans to preprocess.
        training_folder_path (str, optional): The folder that will store training scans. Defaults to None.
        ideal_folder_path (str, optional): The folder that will store ideal or ground truth scans. Defaults to None.
    """
    def __init__(self, input_folder_path, training_folder_path=None, ideal_folder_path=None) -> None:
        """
        Initializes DatasetPreparation object.

        Args:
            input_folder_path (str): The path to the input dataset folder.
            training_folder_path (str, optional): The path to the folder that stores the training scans. Defaults to None.
            ideal_folder_path (str, optional): The path to the folder that will store the ideal scans. Defaults to None.
        """
        self.input_folder_path = input_folder_path
        if training_folder_path != None:
            self.training_folder_path = training_folder_path
        if ideal_folder_path != None:
            super().__init__(self.input_folder_path, ideal_folder_path)
    
    def create_training_data(self):
        """
        Creates training data by applying random bias field, random motion, and random noise to input scans.
        """
        transforms = tio.transforms.Compose([tio.transforms.RandomBiasField(coefficients=1, order=3), 
                                             tio.transforms.RandomMotion(degrees=4, translation=4, num_transforms=2), 
                                             tio.transforms.RandomNoise(mean=2, std=1)])
        for i in tqdm(self.input_files):
            image = tio.ScalarImage(f"{self.input_folder_path}/{i}")
            augmented_image = transforms(image)
            augmented_image.save(f"{self.output_folder_path}/{i}")
    
    def baselineImprov(self):
        """
        Creates ideal data by correcting bias fields, conducting min-max normalization, and removing noise. 
        """
        denoise = sitk.MinMaxCurvatureFlowImageFilter()
        for i in tqdm(self.input_files):
            image_path = os.listdir(f"{super().input_folder}/{i}/anat")
            image = sitk.ReadImage(f"{super().input_folder}/{i}/anat/{image_path}", sitk.sitkFloat32)
            image_mask = super().createMRIMask(image_path, image_created=True)
            bias_corrected_image = super().correctBias(input_image_path=None, image_created=True, image=image, image_mask=image_mask)
            denoised_image = denoise.Execute(bias_corrected_image)
            print("Image denoised!")
            image_array = sitk.GetArrayFromImage(denoised_image)
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
            normalized_image = super().minMaxNormalization(image_tensor)
            print("Image normalized!")
            norm_img_array = np.array(normalized_image)
            norm_img_nifti = nib.Nifti1Image(norm_img_array, affine=np.eye(4))
            nib.save(norm_img_nifti, f"{super().output_folder}/{i}")
            print("Enhancement Successful!")


