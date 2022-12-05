import torchio as tio
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu


class Preprocessing:

    def prepareImgs(self):
        pass

    def twoScannered(self):
        directory = "training"
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            bias = tio.RandomBiasField(coefficients=(0, 1), order=1)
            blur = tio.RandomBlur(std=(0, 1))
            ras = tio.ToCanonical()

            temp_img_1 = tio.ScalarImage(f)
            temp_img_2 = ras(temp_img_1)
            temp_img_3 = blur(temp_img_2)
            temp_img_4 = bias(temp_img_3)
            new_path = f.replace("scanner1", "scanner2")

    def correctBias(self, input_image_path):
        input_image = sitk.ReadImage(input_image_path, sitk.sitkFloat32)
        maskImage = sitk.ReadImage("temp.nii", sitk.sitkUInt8)
        os.remove("temp.nii")
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

        sitk.WriteImage(corrected_image_full_resolution, "cixed.nii")

    def createMRIMask(self, input_image_path):
        img = nib.load(input_image_path)
        data = np.squeeze(img.get_fdata())
        b0_mask, mask = median_otsu(data, median_radius=2, numpass=1)
        mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
        nib.save(mask_img, "temp.nii")
