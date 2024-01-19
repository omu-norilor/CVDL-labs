import os
import cv2
import wget
import glob
import wandb
import shutil
import numpy as np

import torch
import wandb
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from dataset import LFWDataset
from tqdm import tqdm

#fix random seed
np.random.seed(42)
torch.manual_seed(42)


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
    
# class DiceLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, target):
#         smooth = 1.0
#         input_flat = input.reshape(-1)
#         target_flat = target.reshape(-1)
#         intersection = torch.sum(input_flat * target_flat)
#         dice = (2.0 * intersection + smooth) / (torch.sum(input_flat) + torch.sum(target_flat) + smooth)
#         return 1.0 - dice
    
class DiceLoss(torch.nn.Module):
    def init(self):
        super(DiceLoss, self).init()
        
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    
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
            plt.imshow(inputs[j].cpu().numpy().astype(np.uint8))
            plt.subplot(2, 5, j + 6)
            plt.imshow(targets[j].cpu().numpy().astype(np.uint8))
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
    # init wandb
    wandb.init(project="your_project_name", name="training_run")
    
    dataset = LFWDataset(base_folder='data', transforms=None, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Instantiate the UNet model
    model = UNet(retain_dim=True, out_sz=(250,250),num_class=3)

    # Define the loss function and optimizer
    criterion = DiceLoss()

    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 3e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    # early_stopping = EarlyStopping(patience=5, verbose=True)

    model=model.cuda()

    # Training loop1
    num_epochs = 10  # Adjust the number of epochs as needed
    model.train()
    print('Starting Training')
    for epoch in range(num_epochs):
        total_loss = 0.0

        # Wrap the dataloader with tqdm to display a progress bar
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss    
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Print average loss for the epoch
        average_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
        # Log metrics to wandb
        wandb.log({"Epoch": epoch + 1, "Train Loss": average_loss, "Learning Rate": scheduler.get_last_lr()[0]})

        
        # # if new minimum loss is found, save model
        # if total_loss < early_stopping.val_loss_min:
        #     torch.save(model.state_dict(), 'unet.pth')
            
        #  # Check for early stopping
        # if early_stopping.monitor_step(average_loss):
        #     print(f'Early stopping at epoch {epoch+1}')
        #     break


    print('Training complete!')

    # Save the model
    torch.save(model.state_dict(), 'unet.pth')

def test_prediction():
    wandb.init(project="your_project_name", name="test_predictions")

    # load dataset
    dataset = LFWDataset(base_folder='data', transforms=None, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # load model from file
    model = UNet(retain_dim=True, out_sz=(250,250),num_class=3)
    model.load_state_dict(torch.load('unet.pth'))
    model=model.cuda()

    # make some predictions and show them
    i=0
    n=5
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)

            outputs = torch.argmax(outputs, dim=1)
            wandb_table_data = []
            wandb_table = wandb.Table(data=wandb_table_data, columns=["Input Image", "Predicted Image"])
            wandb.log({"Predictions": wandb_table})
            for j in range(8):

                plt.subplot(3, 8, j + 1)
                plt.imshow(inputs[j].cpu().permute(1, 2, 0).numpy().astype(np.uint8))
                plt.title("Input")

                plt.subplot(3, 8, j + 9)
                plt.imshow(targets[j].cpu().permute(1, 2, 0).numpy().astype(np.uint8))
                plt.title("Target")

                # convert outputs from segmentation mask to RGB
                matrix= outputs[j].cpu().numpy()
                

                rgb_image = np.zeros((250, 250, 3), dtype=np.uint8)

                for channel in range(3):
                        # Set the values in the channel to 255 where the outputs values match the channel number
                        rgb_image[:, :, channel] = (matrix == channel + 1) * 255

                rgb_image = rgb_image[:, :, (2, 0, 1)]
                plt.subplot(3, 8, j + 17)
                plt.imshow(rgb_image.astype(np.uint8))
                plt.title("Output")
                wandb_table_data.append([wandb.Image(targets[j].cpu().permute(1, 2, 0).numpy().astype(np.uint8)),
                                    wandb.Image(rgb_image)])
            plt.show()

            i+=1
            if i==n:
                break

# ==================== #
#     Main  region     #
# ==================== #

if __name__ == '__main__':

    # TESTS
    # test_dataset()
    # test_block()
    # test_encoder()
    # test_decoder()
    # test_unet()

    # TRAINING
    # unet_training_loop()
    test_prediction()