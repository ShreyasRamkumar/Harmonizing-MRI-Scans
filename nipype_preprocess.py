from nipype import Node, Workflow, Function
from nipype.interfaces.spm import Realign
from nipype.interfaces.ants import N4BiasFieldCorrection
from os.path import abspath
import torch

def minMaxNormalization(image):
    tensor_min = torch.min(image)
    tensor_max = torch.max(image)
    normalized_image = (image - tensor_min) / (tensor_max - tensor_min)
    return normalized_image

in_file = abspath("raw.nii")

bias = Node(N4BiasFieldCorrection(in_file=in_file, dimension=3, bspline_fitting_distance=300, shrink_factor=3))
motion = Node(Realign(in_file=bias.outputs.out_file, register_to_mean=True))

wf = Workflow(name="preprocessing")
wf.connect([bias, motion, [("out_file", "in_file")]])