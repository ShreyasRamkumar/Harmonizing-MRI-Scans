from torch import optim, nn, ones, sqrt, mean

class Network_Utility:
    @staticmethod
    def ssim(img1, img2, window_size=11, sigma=1.5):
        # Constants for numerical stability
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Data normalization
        mean1 = nn.conv2d(img1, ones(1, 1, window_size, window_size) / window_size ** 2, padding=window_size // 2)
        mean2 = nn.conv2d(img2, ones(1, 1, window_size, window_size) / window_size ** 2, padding=window_size // 2)
        mean_sq1 = mean1 ** 2
        mean_sq2 = mean2 ** 2
        mean12 = mean1 * mean2

        var1 = nn.conv2d(img1 ** 2, ones(1, 1, window_size, window_size) / window_size ** 2, padding=window_size // 2) - mean_sq1
        var2 = nn.conv2d(img2 ** 2, ones(1, 1, window_size, window_size) / window_size ** 2, padding=window_size // 2) - mean_sq2
        covar12 = nn.conv2d(img1 * img2, ones(1, 1, window_size, window_size) / window_size ** 2, padding=window_size // 2) - mean12

        # SSIM components
        luminance = (2 * mean12 + C1) / (mean_sq1 + mean_sq2 + C1)
        contrast = (2 * sqrt(var1) * sqrt(var2) + C2) / (var1 + var2 + C2)
        structure = (covar12 + C2 / 2) / (sqrt(var1) * sqrt(var2) + C2 / 2)

        # SSIM index
        ssim_index = luminance * contrast * structure

        # Average over spatial dimensions
        ssim_value = mean(ssim_index, dim=(2, 3))

        return 1 / ssim_value

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
    def create_data_splits(dataset_len):
        training_len = int(dataset_len * 0.8)
        validation_len = int((dataset_len - training_len) / 2)
        return [training_len, validation_len, validation_len]