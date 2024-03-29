from numpy.lib.stride_tricks import sliding_window_view
from numpy import argwhere, mod, asarray, mean, std, eye
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import shannon_entropy
import torch

class Image:
    def __init__(self, x_image, y_image, id):
        self.id = id
        self.nifti = {"x": nib.load(x_image), "y_hat": 0, "y": nib.load(y_image)} # input, predicted, ground truth
        self.slice_index = 0
        self.slices = {"x": 0, "y_hat": 0, "y": 0} # input, predicted, ground truth
        self.get_slice()
        self.slices.update({"y": self.nifti["y"].get_fdata()[:, :, self.slice_index]})

        # self.cnr = [self.calculate_cnr(self.slices["x"]), 0, self.calculate_cnr(self.slices["y"])], # input, predicted, ground truth

    def get_slice(self):
        scan_entropies = []
        scan_array = self.nifti["x"].get_fdata()
        for i in tqdm(range(192)):
            scan_slice = scan_array[:, :, i]
            entropy = shannon_entropy(scan_slice)
            scan_entropies.append(entropy)
        max_entropy = max(scan_entropies)
        self.slice_index = scan_entropies.index(max_entropy)
        self.slices["x"] = scan_array[:, :, self.slice_index]

    def update_cnr(self):
        try:
            yhat_cnr = self.calculate_cnr(self.slices[1])
        except:
            yhat_cnr = 0
        self.cnr = [self.calculate_cnr(self.slices[0]), yhat_cnr, self.calculate_cnr(self.slices[2])], # input, predicted, ground truth

    def calculate_cnr(self, slice):
        gm = mod(slice, 0.6)
        wm = mod(slice, 0.8)
        gm[gm < 0.445] = 0
        wm[wm < 0.50] = 0

        n1_indices = [i[1] for i in self.define_n1(gm, 0.4)]
        n2_indices = self.define_n2(wm, 0.6)[1]

        self.n_1 = asarray([slice[i[0]:i[0]+2, i[1]:i[1]+2] for i in n1_indices])
        self.n_2 = asarray(slice[n2_indices[0]:n2_indices[0]+8, n2_indices[1]:n2_indices[1]+8])
        
        cnr = (mean(self.n_2, axis=None) - mean(self.n_1, axis=None)) / (std(self.n_2, axis=None) - std(self.n_1, axis=None))
        return cnr

    def define_n1(self, matrix, target_average):
        rows, cols = matrix.shape
        n_1_kernels = []
        kernel_shape = (2, 2)
        mid_row = rows // 2
        mid_col = cols // 2

        submatrix_1 = matrix[:mid_row, :mid_col]
        submatrix_2 = matrix[:mid_row, mid_col:]
        submatrix_3 = matrix[mid_row:, :mid_col]
        submatrix_4 = matrix[mid_row:, mid_col:]

        submatrices = [submatrix_1, submatrix_2, submatrix_3, submatrix_4]

        for i in range(4):
            overlapping_submatrices = sliding_window_view(submatrices[i], kernel_shape)
            averages = overlapping_submatrices.mean(axis=(2, 3))
            mask = averages >= target_average
            indices = argwhere(mask)
            
            if len(indices) > 0:
                j, k = indices[0]
                if i == 1:
                    k += mid_col
                elif i == 2:
                    j += mid_row
                elif i == 3:
                    j += mid_row
                    k += mid_col

                submatrix = matrix[j:j+2, k:k+2]    
                n_1_kernels.append([submatrix, (j, k)])
            else:
                return None, None, None

        return n_1_kernels
    
    def define_n2(self, matrix, target_average):
        submatrix_shape = (8,8)
        overlapping_submatrices = sliding_window_view(matrix, submatrix_shape)
        averages = overlapping_submatrices.mean(axis=(2, 3))
        mask = averages >= target_average
        indices = argwhere(mask)
        if len(indices) > 0:
            i, j = indices[0]
            n_2 = matrix[i:i+8, j:j+8]    
            return n_2, (i, j), n_2.shape
        else:
            return None, None, None
    
    def save_slice(self, slice_index, directory):
        slice_nifti = nib.Nifti1Image(self.slices[slice_index], affine=eye(4))
        nib.save(slice_nifti, directory+self.id+".nii")

    def add_slice(self, slice_index, slice):
        self.slices[slice_index] = slice
        self.update_cnr()


if __name__ == "__main__":
    image = Image(x_image="./data/processed_input/sub-CC610022/anat/sub-CC610022.nii", y_image="./data/ground_truth/sub-CC610022/anat/sub-CC610022.nii", id='a')