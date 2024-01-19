import torch

# Create a conv layer with 32 output channels and a kernel size of 3x3
conv = torch.nn.Conv2d(3, 32, 3)

# Pass an input image to the conv layer
input = torch.randn(1, 3, 224, 224)
output = conv(input)

# Print the output feature maps
print(output.shape)