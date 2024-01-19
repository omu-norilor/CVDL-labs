import os
import cv2
import wget
import glob
import wandb
import shutil
import numpy as np

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from dataset import LFWDataset
from tqdm import tqdm

# ==================== #
#  Architecure region  #
# ==================== #

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
    
class Decoder(torch.nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = torch.nn.ModuleList([torch.nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = torch.nn.ModuleList([EncoderBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_filters = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_filters], dim=1)
            x = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_filters, x):
        _, _, H, W = x.shape
        enc_filters = torchvision.transforms.CenterCrop([H, W])(enc_filters)
        return enc_filters


class UNet(torch.nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
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


class MeanPixelAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        assert input.shape == target.shape, "Input and target shapes must match"
        # Convert the input and target to LongTensor to use them as indices
        input = input.long()
        target = target.long()

        # Calculate mean pixel accuracy
        correct_pixels = torch.sum(input == target)
        total_pixels = input.numel()
        mean_pixel_accuracy = correct_pixels.float() / total_pixels
        return mean_pixel_accuracy

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


# ==================== #
#    Testing region    #
# ==================== #

def test_dataset():
    dataset = LFWDataset(base_folder='data', transforms=None, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    i = 0
    n = 5 
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print('inputs shape: ', inputs.shape, ' targets shape: ', targets.shape)
        # plot images
        plt.figure(figsize=(10, 10))
        for j in range(5):
            import pdb; pdb.set_trace()
            plt.subplot(2, 5, j + 1)
            plt.imshow(cv2.cvtColor(inputs[j].cpu().numpy() , cv2.COLOR_BGR2RGB))
            plt.subplot(2, 5, j + 6)
            plt.imshow(cv2.cvtColor(targets[j].cpu().numpy(), cv2.COLOR_BGR2RGB))
        plt.show()
        
        i += 1
        if i == n:
            break
  
def test_block():
    enc_block = EncoderBlock(1, 64)
    x = torch.randn(32, 3, 572, 572)
    print(enc_block(x).shape)

def test_encoder():
    encoder = Encoder()
    # input image
    x  = torch.randn(1, 3, 572, 572)
    ftrs = encoder(x)
    for ftr in ftrs: 
        print(ftr.shape)

def test_decoder():
    encoder = Encoder()
    # input image
    x  = torch.randn(1, 3, 572, 572)
    filters = encoder(x)
    decoder = Decoder()
    x = torch.randn(1, 1024, 28, 28)
    print(decoder(x, filters[::-1][1:]).shape)

def test_unet():
    unet = UNet()
    x = torch.randn(1, 3, 572, 572)
    print(unet(x).shape)

# ==================== #
#    Training region   #
# ==================== # 

def unet_training_loop():
    
    
    dataset = LFWDataset(base_folder='data', transforms=None, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Instantiate the UNet model
    model = UNet(retain_dim=True, out_sz=(250,250),num_class=3)

    # Define the loss function and optimizer
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 3e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5,6,7,8,9,10,11,13,15], gamma=0.75)

    model=model.cuda()

    # Training loop
    num_epochs = 10  # Adjust the number of epochs as needed

    print('Starting Training')
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Wrap the dataloader with tqdm to display a progress bar
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            # Zero the gradients
            optimizer.zero_grad()

            # reorder dimensions
            # inputs = inputs.permute(0, 3, 1, 2).requires_grad_()
            # targets = targets.permute(0, 3, 1, 2).requires_grad_()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        average_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    print('Training complete!')

    # Save the model
    torch.save(model.state_dict(), 'unet.pth')


# ==================== #
#     Main  region     #
# ==================== #

if __name__ == '__main__':

    # TESTS
    test_dataset()
    # test_block()
    # test_encoder()
    # test_decoder()
    # test_unet()

    # TRAINING
    # unet_training_loop()