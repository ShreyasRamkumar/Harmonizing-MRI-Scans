from nipype import Node, Workflow, Function
from nipype.interfaces.spm import Realign
from nipype.interfaces.ants import N4BiasFieldCorrection
from os.path import abspath
import torch
import nibabel as nib 
import numpy as np

def minMaxNormalization(image):
    tensor_min = torch.min(image)
    tensor_max = torch.max(image)
    normalized_image = (image - tensor_min) / (tensor_max - tensor_min)
    return normalized_image

def wf(input_image):
    img = nib.load(input_image)
    img_array = img.get_fdata()
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    normalize = Node(Function(input_names=["image"], output_names=["normalized_image"], function=minMaxNormalization), name="min_max_norm")
    normalize.inputs.image=img_tensor
    normalize.run()
    normalized_image_array = normalize.outputs.normalized_image.numpy()
    normalized_image = nib.Nifti1Image(normalized_image_array, affine=np.eye(4))
    nib.save(normalized_image, "./preprocessed1.nii")


wf("raw.nii")