import os 
import nibabel as nib
from torch import nn
import torch
import time
import numpy as np

def crop_center(img: np.ndarray) -> np.ndarray:
        """
        Pad an image to the specified shape, centered around the center of the image.
        """
        x, y, z = img.shape
        pad_x, pad_y, pad_z = 224, 192, 384
        x_pad = max(pad_x - x, 0)
        y_pad = max(pad_y - y, 0)
        z_pad = max(pad_z - z, 0)
        x_start = x_pad // 2
        y_start = y_pad // 2
        z_start = z_pad // 2
        x_end = x_start + x
        y_end = y_start + y
        z_end = z_start + z
        padded_img = np.zeros((pad_x, pad_y, pad_z), dtype=img.dtype)
        padded_img[x_start:x_end, y_start:y_end, z_start:z_end] = img
        return padded_img

def uncrop_center(img: np.ndarray):
    """
    Uncrop a padded image back to its original shape.
    """
    pad_x, pad_y, pad_z = img.shape
    x, y, z = 224, 174, 370
    x_start = (pad_x - x) // 2
    y_start = (pad_y - y) // 2
    z_start = (pad_z - z) // 2
    x_end = x_start + x
    y_end = y_start + y
    z_end = z_start + z
    uncropped_img = img[x_start:x_end, y_start:y_end, z_start:z_end]
    return uncropped_img


def torch_fws_input_data(ip: np.ndarray, op: np.ndarray, denom: float = 1024) -> (np.ndarray, np.ndarray):
    inp = torch.from_numpy(np.stack((crop_center(ip), crop_center(op)))).float()
    return inp

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)


    def forward(self, x, skip_input):
        x = torch.cat((x, skip_input), 1)
        x = self.model(x)
        x =  nn.functional.pad(x, (1,0,1,0,1,0))

        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.mid1 = UNetMid(1024, 512, dropout=0.2)
        self.mid2 = UNetMid(1024, 512, dropout=0.2)
        self.mid3 = UNetMid(1024, 512, dropout=0.2)
        self.mid4 = UNetMid(1024, 256, dropout=0.2)
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose3d(128, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m1 = self.mid1(d4, d4)
        m2 = self.mid2(m1, m1)
        m3 = self.mid3(m2, m2)
        m4 = self.mid4(m3, m3)
        u1 = self.up1(m4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)


def fat_water_separation_torch(model_path: str, ip_path: str, op_path: str, denom: float = 1024) -> None:

    if not os.path.exists(model_path):
        print("Model path does not exist.", model_path)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()

    start_time = time.time()
    ip = nib.load(ip_path)
    ip_data = np.asanyarray(nib.load(ip_path).dataobj) / denom
    op_data = np.asanyarray(nib.load(op_path).dataobj) / denom
    elapsed_time = time.time() - start_time
    print("data loading time:", elapsed_time, "seconds")
    start_time = time.time()
    input_data = torch_fws_input_data(ip_data, op_data)
    input_data_array = input_data.numpy()  # Convert input_data Tensor to a NumPy array
    input_data_tensor = torch.from_numpy(input_data_array).unsqueeze(0).float().to(device)
    elapsed_time = time.time() - start_time
    print("input processing time:", elapsed_time, "seconds")

    # Perform fat-water separation
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_data_tensor)
        elapsed_time = time.time() - start_time
        print("inference time:", elapsed_time, "seconds")
        fat_prediction, water_prediction = torch.split(outputs, 1, dim=1)
        fat_prediction = fat_prediction.squeeze(0).squeeze(0).cpu().numpy()
        water_prediction = water_prediction.squeeze(0).squeeze(0).cpu().numpy()
    start_time = time.time()
    fat_prediction = uncrop_center(fat_prediction)
    water_prediction = uncrop_center(water_prediction)
    # Scale and save the predictions
    fat_nii = nib.Nifti1Image((fat_prediction * denom).astype('uint16'), ip.affine, ip.header)
    fat_nii.to_filename('fat_prediction.nii.gz')
    water_nii = nib.Nifti1Image((water_prediction * denom).astype('uint16'), ip.affine, ip.header)
    water_nii.to_filename('water_prediction.nii.gz')
    elapsed_time = time.time() - start_time
    print("finish/saving time:", elapsed_time, "seconds")

    return


def fat_water_separation(factor: float = 1024.):
    fat_water_separation_torch(model_path, ip_path, op_path, factor)
