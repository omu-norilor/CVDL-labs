
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from dataset import LFWDataset
from tqdm import tqdm



class EncoderBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(torch.nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = torch.nn.ModuleList([EncoderBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = torch.nn.MaxPool2d(2)
    
    def forward(self, x):
        filters = []
        for block in self.enc_blocks:
            x = block(x)
            filters.append(x)
            x = self.pool(x)
        return filters
    
class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        _, _, h, w = x.size()
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return x[:, :, i:(i+th), j:(j+tw)]


class Decoder(torch.nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = torch.nn.ModuleList([torch.nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = torch.nn.ModuleList([EncoderBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        enc_list = []
        for i, upconv in enumerate(self.upconvs):
            x = upconv(x)
            enc_filters = self.crop(encoder_features[i], x)
            enc_list.append(enc_filters)

        enc_list = enc_list[::-1]
        for i, decblock in enumerate(self.dec_blocks):
            enc_filters = enc_list.pop()
            x = torch.cat([x, enc_filters], dim=1)
            x = decblock(x)
        return x
    
    # def crop(self, enc_filters, x):
    #     _, _, H, W = x.shape
    #     enc_filters = torchvision.transforms.CenterCrop([H, W])(enc_filters)
    #     return enc_filters

    def crop(self, enc_filters, x):
        _, _, H, W = x.shape
        h, w = enc_filters.shape[2:]

        # Calculate the crop boundaries
        crop_h_start = (h - H) // 2
        crop_w_start = (w - W) // 2

        # Perform the center crop
        enc_filters = enc_filters[:, :, crop_h_start:(crop_h_start + H), crop_w_start:(crop_w_start + W)]

        return enc_filters



class UNet(torch.nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=3, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = torch.nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = torch.nn.functional.interpolate(out, self.out_sz)
        return out
    
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.0
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = torch.sum(input_flat * target_flat)
        dice = (2.0 * intersection + smooth) / (torch.sum(input_flat) + torch.sum(target_flat) + smooth)
        return 1.0 - dice

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def monitor_step(self, val_loss):
        # make val_loss absolute so that it is always positive
        val_loss = abs(val_loss)
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score and score is not np.nan:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss
