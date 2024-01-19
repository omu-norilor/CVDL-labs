

def upsample_block(x, filters, size, stride = 2):
    """
    x - the input of the upsample block
    filters - the number of filters to be applied
    size - the size of the filters
    """

    # TODO your code here
    # transposed convolution
    # import pdb; pdb.set_trace()
    conv= torch.nn.ConvTranspose2d(in_channels=x.shape[1], out_channels=filters, kernel_size=size, stride=stride, padding=0)

    x = conv(x)
    
    # BN
    x = torch.nn.BatchNorm2d(filters)(x)
    
    # relu activation
    x = torch.nn.ReLU()(x)

    return x

def test_upsample_block():
    in_layer = torch.rand((32, 32, 128, 128))

    filter_sz = 4
    num_filters = 16

    for stride in [2, 4, 8]:
        x = upsample_block(in_layer, num_filters, filter_sz, stride)
        print('in shape: ', in_layer.shape, ' upsample with filter size ', filter_sz, '; stride ', stride, ' -> out shape ', x.shape)
    
# class Encoder(torch.nn.Module):
#     def __init__(self, channels_list, num_blocks,image_size=250):
#         super(Encoder, self).__init__()

#         self.channels_list = channels_list
#         self.image_size = image_size
#         self.num_blocks = num_blocks

#         # Initialize blocks based on the number of blocks
#         for i in range(num_blocks):
#             # Define convolutional layers for each 
#             #channels_list = [3, 64, 128, 256]
#             setattr(self, f'conv{i}1', torch.nn.Conv2d(channels_list[i], channels_list[i+1], kernel_size=3, padding=0, dtype=torch.float))
#             setattr(self, f'relu{i}', torch.nn.ReLU())
#             setattr(self, f'conv{i}2', torch.nn.Conv2d(channels_list[i+1], channels_list[i+1], kernel_size=3, padding=0, dtype=torch.float))

#             # Define max pooling layer for each block (except the last one)
#             if i < num_blocks - 1:
#                 setattr(self, f'maxpool{i}', torch.nn.MaxPool2d(2, stride=2))

#     def forward(self, x):
#         outputs = []

#         for i in range(self.num_blocks):
#             # Convolutional layers for each block
#             x = getattr(self, f'conv{i}1')(x)
#             x = getattr(self, f'relu{i}')(x)
#             x = getattr(self, f'conv{i}2')(x)
#             outputs.append(x)

#             # Max pooling for each block (except the last one)
#             if i < self.num_blocks - 1:
#                 x = getattr(self, f'maxpool{i}')(x)

#         return outputs
    

# class Decoder(torch.nn.Module):
#     def __init__(self, depth_list, num_blocks):
#         super(Decoder, self).__init__()

#         self.num_blocks = num_blocks
#         self.depth_list = depth_list

#         # Initialize blocks based on the number of blocks
#         for i in range(num_blocks):
            
#             # upsample block
#             upsample_block = torch.nn.ConvTranspose2d(depth_list[i], depth_list[i+1], kernel_size=4, stride=2, dtype=torch.float)
#             setattr(self, f"upsample{i}", upsample_block)
#             # batch norm
#             setattr(self, f'bn{i}', torch.nn.BatchNorm2d(depth_list[i+1]))
#             # relu
#             setattr(self, f'relu{i}1', torch.nn.ReLU())
#             # conv
#             setattr(self, f'conv{i}1', torch.nn.Conv2d(depth_list[i+1], depth_list[i+1], kernel_size=3, padding=0, dtype=torch.float))
#             # relu
#             setattr(self, f'relu{i}2', torch.nn.ReLU())
#             # conv
#             setattr(self, f'conv{i}2', torch.nn.Conv2d(depth_list[i+1], depth_list[i+1], kernel_size=3, padding=0, dtype=torch.float))
            

#     def forward(self, x, encoder_activation_list):
#         for i in range(self.num_blocks):
#             # Upsampling operation
#             upsample_block = getattr(self, f"upsample{i}")
#             x = upsample_block(x)
            
#             # Apply the convolutional layers for each block
#             bn = getattr(self, f'bn{i}')
#             relu1 = getattr(self, f'relu{i}1')
#             conv1 = getattr(self, f'conv{i}1')
#             relu2 = getattr(self, f'relu{i}2')
#             conv2 = getattr(self, f'conv{i}2')

#             x = bn(x)
#             x = relu1(x)

#             # Crop and concatenate with skip connection from the encoder
#             if i < len(encoder_activation_list):
#                 encoder_activation = encoder_activation_list[i]
#                 crop_size = (x.shape[2], x.shape[3])
#                 encoder_activation = TF.center_crop(encoder_activation, crop_size)
#                 x = torch.cat([x, encoder_activation], dim=1)
                
#             # encoder block
#             x = conv1(x)
#             x = relu2(x)
#             x = conv2(x)

#         return x


# class UNet(torch.nn.Module):
#     def __init__(self, channels_list, depth_list, num_blocks):
#         super(UNet, self).__init__()

#         self.num_blocks = num_blocks
#         self.channels_list = channels_list
#         self.depth_list = depth_list

#         # Initialize the encoder and decoder
#         self.encoder = Encoder(channels_list, num_blocks)
#         self.decoder = Decoder(depth_list, num_blocks)

#         # Initialize the final convolutional layer
#         self.final_conv = torch.nn.Conv2d(depth_list[-1], 1, kernel_size=1, padding=0, dtype=torch.float)

#     def forward(self, x):
#         # Encoder
#         encoder_activation_list = self.encoder(x)

#         reversed_encoder_activation_list = encoder_activation_list[::-1]
#         # Decoder
#         x = self.decoder(encoder_activation_list[-1], reversed_encoder_activation_list)

#         # Final convolutional layer
#         x = self.final_conv(x)

#         # Resize the output to the original size
#         x = torch.nn.functional.interpolate(x,self.image_size,self.image_size)

#         return x