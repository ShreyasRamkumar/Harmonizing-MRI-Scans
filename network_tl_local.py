import os
import lightning.pytorch as pl
from torch import optim, nn
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
from tqdm import tqdm
import skimage
from lightning.pytorch.callbacks import Callback
 
# important folders
model_input_path = "./data/processed_input"
ground_truths_path = "./data/original/"


class Unet_Utility:

    @staticmethod
    def convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_c)
        )
        return run
    
    @staticmethod
    def down_convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_c)
        )
        return run

    @staticmethod
    def up_convolution(in_c, out_c):
        run = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_c),
        )
        return run

    @staticmethod
    def final_convolution(in_c, out_c):
        run = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1),
            nn.ReLU()
        )
        return run

    @staticmethod
    def crop_tensor(target_tensor, tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2

        return tensor[:, :, delta:tensor_size- delta, delta:tensor_size-delta]

    @staticmethod
    def get_slice(scan_tensor):
        scan_entropies = []
        for i in tqdm(range(256)):
            scan_slice = scan_tensor[:, :, i]
            entropy = skimage.measure.shannon_entropy(scan_slice)
            scan_entropies.append(entropy)
        max_entropy = max(scan_entropies)
        max_entropy_slice_index = scan_entropies.index(max_entropy)
        return max_entropy_slice_index
    
    def create_data_splits(dataset_len):
        training_len = int(dataset_len * 0.8)
        validation_len = int((dataset_len - training_len) / 2)
        return [training_len, validation_len, validation_len]

class SaveOutput(Callback):
    def on_test_end(self, trainer, pl_module):
        outputs = pl_module.testing_outputs
        y_hat_directory = "./data/postprocessed"
        
        for i in range(len(outputs)):
            y_hat_path = f"{y_hat_directory}/{i}.nii"
            cpu_tensor = outputs[i].cpu()
            y_hat_array = cpu_tensor.numpy()
            y_hat_scan = nib.Nifti1Image(y_hat_array, affine=np.eye(4))
            nib.save(y_hat_scan, y_hat_path)
        

# Model Class
class Unet(pl.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        # hyperparameters
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.testing_outputs = []
        self.validation_outputs = []
        
        # definition of neural network 

        # naming convention = o_number of channels_encode/decode_up/down?side
        self.o_16_encode_side = Unet_Utility.convolution(1, 16)
        self.o_16_encode_down = Unet_Utility.down_convolution(16, 16)
        self.o_32_encode_side = Unet_Utility.convolution(16, 32)
        self.o_32_encode_down = Unet_Utility.down_convolution(32, 32)
        self.o_64_encode_side = Unet_Utility.convolution(32, 64)
        self.o_64_encode_down = Unet_Utility.down_convolution(64, 64)
        self.o_128_encode_side = Unet_Utility.convolution(64, 128)
        self.o_128_encode_down = Unet_Utility.down_convolution(128, 128)
        self.o_256_encode_side = Unet_Utility.convolution(128, 256)
        self.o_128_decode_up = Unet_Utility.up_convolution(256, 128)
        self.o_128_decode_side = Unet_Utility.convolution(256, 128)
        self.o_64_decode_up = Unet_Utility.up_convolution(128, 64)
        self.o_64_decode_side = Unet_Utility.convolution(128, 64)
        self.o_32_decode_up = Unet_Utility.up_convolution(64, 32)
        self.o_32_decode_side = Unet_Utility.convolution(64, 32)
        self.o_16_decode_up = Unet_Utility.up_convolution(32, 16)
        self.o_16_decode_side = Unet_Utility.convolution(32, 16)
        self.o_1_decode_side = Unet_Utility.final_convolution(17, 1)
    
    # forward pass
    def forward(self, image):
        # naming convention: x_number of channels_encode/decode_up/down/nothing(side convolution)
        
        x_16_encode = self.o_16_encode_side(image) # has a shape of [1, 16, 256, 256]
        x_16_encode_down = self.o_16_encode_down(x_16_encode) # has a shape of [1, 16, 128, 128]
        x_32_encode = self.o_32_encode_side(x_16_encode_down) # has a shape of [1, 32, 128, 128]
        x_32_encode_down = self.o_32_encode_down(x_32_encode) # has a shape of [1, 32, 64, 64]
        x_64_encode = self.o_64_encode_side(x_32_encode_down) # has a shape of [1, 64, 64, 64]     
        x_64_encode_down = self.o_64_encode_down(x_64_encode) # has a shape of [1, 64, 32, 32]
        x_128_encode = self.o_128_encode_side(x_64_encode_down)  # has a shape of [1, 128, 32, 32]      
        x_128_encode_down = self.o_128_encode_down(x_128_encode) # has a shape of [1, 128, 16, 16]
        x_256_encode = self.o_256_encode_side(x_128_encode_down) # has a shape of [1, 256, 16, 16]

        x_256_decode_up = self.o_128_decode_up(x_256_encode) # has a shape of [1, 128, 22, 22]
        x_256_decode_cat = torch.cat([x_256_decode_up, x_128_encode], 1) # has a shape of [1, 256, 32, 32]       
        x_128_decode = self.o_128_decode_side(x_256_decode_cat) # has a shape of [1, 128, 32, 32]
        x_128_decode_up = self.o_64_decode_up(x_128_decode) # has a shape of [1, 64, 64, 64]
        x_128_decode_cat = torch.cat([x_128_decode_up, x_64_encode], 1) # has a shape of [1, 128, 64, 64]
        x_64_decode = self.o_64_decode_side(x_128_decode_cat) # has a shape of [1, 64, 64, 64]
        x_64_decode_up = self.o_32_decode_up(x_64_decode) # has a shape of [1, 32, 128, 128]
        x_64_decode_cat = torch.cat([x_64_decode_up, x_32_encode], 1) # has a shape of [1, 64, 128, 128]
        x_32_decode = self.o_32_decode_side(x_64_decode_cat) # has a shape of [1, 32, 128, 128]
        x_32_decode_up = self.o_16_decode_up(x_32_decode) # has a shape of [1, 16, 256, 256]
        x_32_decode_cat = torch.cat([x_32_decode_up, x_16_encode], 1) # has a shape of [1, 32, 256, 256]
        x_16_decode = self.o_16_decode_side(x_32_decode_cat) # has a shape of [1, 16, 256, 256]
        x_16_decode_cat = torch.cat([x_16_decode, image], 1) # has a shape of [1, 17, 256, 256]
        final_image = self.o_1_decode_side(x_16_decode_cat)

        assert final_image.shape == image.shape
        
        return final_image

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr = self.learning_rate)
        return opt
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch["scan"]
        y = train_batch["ground_truth"]
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x = test_batch["scan"]
        y = test_batch["ground_truth"]
        y_hat = self.forward(x)
        print(y_hat.shape)
        self.testing_outputs.append(y_hat)

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["scan"]
        y = val_batch["ground_truth"]
        y_hat = self.forward(x)

        self.validation_outputs.append(y_hat)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
    

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.training = []
        self.ground_truth_training = []
        self.testing = []
        self.ground_truth_testing = []
        self.validation = []
        self.ground_truth_validation = []
        self.input_files = os.listdir(model_input_path)
        self.ground_truth_files = os.listdir(ground_truths_path)

    def setup(self, stage: str):
        # set up training, testing, validation split
        
        lens = Unet_Utility.create_data_splits(len(self.input_files))
        training_stop_index = lens[0]
        testing_stop_index = lens[0] + lens[1]
        validation_stop_index = lens[0] + lens[1] + lens[2] - 1

        self.training = self.input_files[:training_stop_index]
        self.ground_truth_training = self.ground_truth_files[:training_stop_index]

        self.training_dataset = MRIDataset(self.training, self.ground_truth_training)
        self.training_dataloader = self.train_dataloader()

        self.testing = self.input_files[training_stop_index:testing_stop_index]
        self.ground_truth_testing = self.ground_truth_files[training_stop_index:testing_stop_index]  

        self.testing_dataset = MRIDataset(self.testing, self.ground_truth_testing)
        self.testing_dataloader = self.test_dataloader()

        self.validation = self.input_files[testing_stop_index:validation_stop_index] 
        self.ground_truth_validation = self.ground_truth_files[testing_stop_index:validation_stop_index]

        self.validation_dataset = MRIDataset(self.validation, self.ground_truth_validation)
        self.validation_dataloader = self.val_dataloader()


    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size = self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.testing_dataset, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size = self.batch_size)

    


class MRIDataset(Dataset):
    def __init__(self, model_input: list = [], ground_truth: list = []):
        super().__init__()
        self.model_input = model_input
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.model_input)

    def __getitem__(self, index):
        scan_path = self.model_input[index]
        intermediate_scan_list = os.listdir(f"{model_input_path}/{scan_path}/anat/")
        scan = nib.load(f"{model_input_path}/{scan_path}/anat/{intermediate_scan_list[0]}")
        scan_array = scan.get_fdata()
        scan_tensor = torch.tensor(scan_array, dtype=torch.float32)
        slice_index = Unet_Utility.get_slice(scan_tensor=scan_tensor)
        scan_slice = scan_tensor[:, :, slice_index]
        scan_slice = scan_slice[None, :, :]

        ground_truth_scan_path = self.ground_truth[index]
        intermediate_scan_list = os.listdir(f"{model_input_path}/{ground_truth_scan_path}/anat/")
        ground_truth_scan = nib.load(f"{model_input_path}/{ground_truth_scan_path}/anat/{intermediate_scan_list[0]}")
        ground_truth_scan_array = ground_truth_scan.get_fdata()
        ground_truth_scan_tensor = torch.tensor(ground_truth_scan_array, dtype=torch.float32)
        ground_truth_scan_slice = ground_truth_scan_tensor[:, :, slice_index]
        ground_truth_scan_slice = ground_truth_scan_slice[None, :, :]

        return {"scan": scan_slice, "ground_truth": ground_truth_scan_slice}


if __name__ == "__main__":
    mri_data = MRIDataModule(batch_size=4)
    model = Unet()
    saveoutput = SaveOutput()
    train = pl.Trainer(max_epochs=200, accelerator="gpu", devices=1, callbacks=[saveoutput])
    # train.fit(model, mri_data)
    train.test(model, mri_data)
        
