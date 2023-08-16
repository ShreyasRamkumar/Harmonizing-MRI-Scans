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

def image_to_tensor(input_image):
    img = nib.load(input_image)
    img_array = img.get_fdata()
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    return img_tensor

def tensor_to_image(input_tensor, output_image_path):
    image_array = input_tensor.numpy()
    image = nib.Nifti1Image(image_array, affine=np.eye(4))
    nib.save(image, output_image_path)

def wf(input_image):
    img_tensor = image_to_tensor(input_image)
    normalize = Node(Function(input_names=["image"], output_names=["normalized_image"], function=minMaxNormalization), name="min_max_norm")
    normalize.inputs.image=img_tensor
    normalize.run()
    tensor_to_image(normalize.outputs.normalized_image, "preprocessed1.nii")

wf("raw.nii")