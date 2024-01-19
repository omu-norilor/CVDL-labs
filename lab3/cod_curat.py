import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# =============================================== #
# ==================== TASKS ==================== #
# =============================================== #

def zero_pad(X, pad):
    """
    This function applies the zero padding operation on all the images in the array X
    :param X input array of images; this array has a of rank 4 (batch_size, height, width, channels)
    :param pad the amount of zeros to be added around around the spatial size of the images
    """
    # hint you might find the function numpy.pad useful for this purpose
    # keep in mind that you only need to pad the spatial dimensions (height and width)
    return np.pad(X, pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)))


def convolution(X, W, bias, pad, stride):
    """
    This function applied to convolution operation on the input X of shape (num_samples, iH, iW, iC)
    using the filters defined by the W (filter weights) and  (bias) parameters.

    :param X - input of shape (num_samples, iH, iW, iC)
    :param W - weights, numpy array of shape (fs, fs, iC, k), where fs is the filter size,
        iC is the depth of the input volume and k is the number of filters applied on the image
    :param biases - numpy array of shape (1, 1, 1, k)
    :param pad - hyperparameter, the amount of padding to be applied
    :param stride - hyperparameter, the stride of the convolution
    """

    # 0. compute the size of the output activation map and initialize it with zeros

    num_samples = X.shape[0]
    iW = X.shape[1] 
    iH = X.shape[2]
    f = W.shape[0]

    # TODO your code here
    # compute the output width (oW), height (oH) and number of channels (oC)
    oH = int((iH - f + 2 * pad) / stride + 1)
    oW = int((iW - f + 2 * pad) / stride + 1)   
    oC = W.shape[3]
    # initialize the output activation map with zeros
    activation_map = np.zeros((num_samples, oH, oW, oC))
    # end TODO your code here

    # 1. pad the samples in the input
    # TODO your code here, pad X using pad amount
    X_padded = zero_pad(X, pad)
    # end TODO your code here
    
    # go through each input sample
    for i in range(num_samples):
        # TODO: get the current sample from the input (use X_padded)
        X_i = X_padded[i]
        # end TODO your code here

        # loop over the spatial dimensions
        for y in range(oH):
            # TODO your code here
            # compute the current ROI in the image on which the filter will be applied (y dimension)
            # tl_y - the y coordinate of the top left corner of the current region
            # br_y - the y coordinate of the bottom right corner of the current region
            tl_y = y * stride
            br_y = y * stride + f
            # end TODO your code here

            for x in range(oW):
                # TODO your code here
                # compute the current ROI in the image on which the filter will be applied (x dimension)
                # tl_x - the x coordinate of the top left corner of the current region
                # br_x - the x coordinate of the bottom right corner of the current region
                tl_x = x * stride
                br_x = x * stride + f
                # end TODO your code here

                for c in range(oC):
                    # select the current ROI on which the filter will be applied
                    roi = X_i[tl_y: br_y, tl_x: br_x, :]
                    w = W[:, :, :, c]
                    b = bias[:, :, :, c]

                    # TODO your code here
                    # apply the filter with the weights w and bias b on the current image roi

                    # Check if roi is not empty before further processing
                    if roi.size == 0:
                        continue

                    # Ensure that the shapes are compatible for element-wise multiplication
                    if roi.shape[2] != w.shape[2]:
                        continue

                    # A. compute the elementwise product between roi and the weights of the filters (np.multiply)
                    a = np.multiply(roi, w)
                    # B. sum across all the elements of a
                    a = np.sum(a)
                    # C. add the bias term
                    a = a + b

                    # D. add the result in the appropriate position of the output activation map
                    activation_map[i, y, x, c] = a
                    # end TODO your code here
                assert (activation_map.shape == (num_samples, oH, oW, oC))
    return activation_map



def download_dataset():
    # TODO you code here
    # - create an object of type torchvision.datasets.OxfordIIITPet, download it
    print("Downloading dataset...")
    transform = transforms.Compose([
        transforms.Resize((388, 500)),  # Adjust the target size to be consistent
        transforms.ToTensor()
        ])
    dataset = datasets.OxfordIIITPet(root="./data",transform=transform, download=True)
    # - torch.utils.data.DataLoader object
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # - display some samples
    for i, (x, y) in enumerate(dataloader):
        if i == 0:
            plt.figure()
            plt.imshow(x[0].permute(1, 2, 0))
            plt.title(y[0])
            plt.show()
            break


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)  # Assuming input image size is 64x64

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)  # Flatten before fully connected layer
        x = self.fc1(x)
        return x



def transfer_training():
    # TODO : your code here
    # get a pretrained torchvision module, change the last layer,  pass a single example through the model

    pretrained_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the last layer to fit your specific task
    num_classes = 37  
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes)

    transform = transforms.Compose([
        transforms.Resize((388, 500)),  # Adjust the target size to be consistent
        transforms.ToTensor()
        ])
    dataset = datasets.OxfordIIITPet(root="./data",transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    class_labels = dataset.classes

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_resnet.to(device)
   
    # optimizer - the chosen optimizer. It holds the current state of the model and will update the parameters based on the computed gradients. Notice that in the constructor of the optimizer you need to pass the parameters of your model and the learning rate.
    # optimizer = torch.optim.Adam(pretrained_resnet.parameters(), lr=0.001)
    learning_rate = 0.001
    optimizer = torch.optim.SGD(pretrained_resnet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # criterion - the chosen loss function.
    criterion = nn.CrossEntropyLoss()
    num_epochs = 10


    for epoch in range(num_epochs):  # num_epochs is a hyperparameter that specifies when is the training process

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0): # iterate over the dataset, now we use data loaders
            # get a batch of data (inputs and their corresponding labels)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU



            # IMPORTANT! set the gradients of the tensors to 0. by default torch accumulates the gradients on subsequent backward passes
            # if you omit this step, the gradient would be a combination of the old gradient, which you have already used to update the parameters
            optimizer.zero_grad()


            # perform the forward pass through the network
            outputs = pretrained_resnet(inputs)
        
            # apply the loss function to determine how your model performed on this batch
            loss = criterion(outputs, labels)

            # start the backprop process. it will compute the gradient of the loss with respect to the graph leaves
            loss.backward()


            # update the model parameters by calling the step function
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[Epoch %d, Batch %5d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    

        #apply the learning rate scheduler 
        scheduler.step()
    return pretrained_resnet, dataset, dataloader, class_labels





# =============================================== #
# ==================== TESTS ==================== #
# =============================================== #

def test_padding():
    img = cv2.imread('cameraman.jpg')
    img = np.asarray([img])
    # img = np.stack([img], axis=3)
    img = zero_pad(img, 100)
    plt.imshow(img[0], cmap='gray', vmin=0, vmax=255)
    plt.show()


def test_conv():
    np.random.seed(10)
    # 100 samples of shape (13, 21, 4)
    X = np.random.randn(100, 13, 21, 4)

    # 8 filters (last dimension) of shape (3, 3)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)

    am = convolution(X, W, b, pad=1, stride=2)
    print("am's mean =\n", np.mean(am))
    print("am[1, 2, 3] =\n", am[3, 2, 1])

def test_lowpass_filters():
    # load the image using Pillow
    image = Image.open('cameraman.jpg')
    image = np.asarray(image)
    # image = np.expand_dims(image, axis=-1)

    # X contains a single image sample
    X = np.expand_dims(image, axis=0)
    
    #############
    # MEAN FILTER
    #############

    bias = np.asarray([0])
    bias = bias.reshape((1, 1, 1, 1))

    mean_filter_3 = np.ones(shape=(3, 3, 1, 1), dtype=np.float32)
    mean_filter_3 = mean_filter_3/9.0

    mean_filter_9 = np.ones(shape=(9, 9, 1, 1), dtype=np.float32)
    mean_filter_9 = mean_filter_9/81.0

    mean_3x3 = convolution(X, mean_filter_3, bias, pad=0, stride=1)
    mean_9x9 = convolution(X, mean_filter_9, bias, pad=0, stride=1)

    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(mean_3x3[0, :, :, 0], cmap='gray')
    plt.title('mean filter 3x3')

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(mean_9x9[0, :, :, 0], cmap='gray')
    plt.title('mean filter 9x9')


    #################
    # GAUSSIAN FILTER
    #################

    gaussian_filter = np.asarray(
        [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]],
        dtype=np.float32
    )
    gaussian_filter = gaussian_filter.reshape(3, 3, 1, 1)
    gaussian_filter = gaussian_filter/16.0

    gaussian_smoothed = convolution(X, gaussian_filter, bias, pad=0, stride=1)

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(gaussian_smoothed[0,:,:,0], cmap='gray')
    plt.title('Gaussian filtered')

    plt.show()

def test_highpass_filters():

    # load the image using Pillow
    image = Image.open('cameraman.jpg')
    image = np.asarray(image)
    # image = np.expand_dims(image, axis=-1)

    # X contains a single image sample
    X = np.expand_dims(image, axis=0)
    bias = np.asarray([0])
    bias = bias.reshape((1, 1, 1, 1))
    sobel_horiz = np.asarray([[-1, 0, 1],
                            [-2, 0, 2], 
                            [-1, 0, 1]])

    sobel_vert = sobel_horiz.T 

    sobel_horiz = np.reshape(sobel_horiz, (3, 3, 1, 1))
    sobel_vert = np.reshape(sobel_vert, (3, 3, 1, 1))

    sobel_x = convolution(X, sobel_horiz, bias, 0, 1)
    sobel_y = convolution(X, sobel_vert, bias, 0, 1)


    plt.subplot(1, 3, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.title('Original image')
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(sobel_x[0,:,:,0])/np.abs(np.max(sobel_x[0,:,:,0]))*255, cmap='gray')
    plt.title('Sobel X')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(sobel_y[0,:,:,0])/np.abs(np.max(sobel_y[0,:,:,0]))*255, cmap='gray')
    plt.title('Sobel Y')
    plt.tight_layout()

def test_transfer_training():
    pretrained_resnet, dataset, dataloader, class_labels = transfer_training()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Predict the class of the first image in the dataset
    for i, (x, y) in enumerate(dataloader):
        if i == 3:
            x, y = x.to(device), y.to(device)  # Move input and label to GPU
            output = pretrained_resnet(x)
            predicted_class_label = class_labels[torch.argmax(output[0]).item()]
            true_class_label = class_labels[y[0].item()]
            print("\nPredicted class: ", predicted_class_label)
            print("True class: ", true_class_label)
            plt.figure()
            plt.imshow(x[0].cpu().permute(1, 2, 0))
            plt.title("Predicted: " + predicted_class_label + " True: " + true_class_label)
            plt.show()


if __name__ == '__main__':     
    # test_padding()
    # test_conv()
    # test_lowpass_filters()
    # test_highpass_filters()
    # download_dataset()
    test_transfer_training()
    pass