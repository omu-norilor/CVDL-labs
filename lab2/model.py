import math
import torch
import numpy as np
from activations import softmax
import torch.nn as nn
from torch.nn.functional import cross_entropy, relu
from torch import optim
import tqdm


class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find torch.randn userful here *0.001
        self.W=np.random.randn(self.input_shape,self.num_classes)*0.001
        # here i set the last row of the weight matrix with 1, such that the bias trick is applied 
        # (the last column was added at the initialization step, so we don't need to do it again,
        #  we just need to set the last row to 1) 
        self.W[-1,:]=1
        self.W=torch.tensor(self.W)
        self.W.requires_grad_()
        # don't forget to set call requires_grad_() on the weight matrix,
        # as we will be be taking its gradients during the learning process

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        # 0. compute the dot product between the input X and the weight matrix
        # you can use @ for this operation
        scores = None
        scores = X.float() @ self.W.float()
        # remember about the bias trick!
        # 1. apply the softmax function on the scores, see torch.nn.functional.softmax
        # think about on what dimension (dim parameter) you should apply this operation
        scores = softmax(scores)
        # 2. returned the normalized scores
        return scores

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        # 0. compute the dot product between the input X and the weight matrix
        scores = None
        scores = X.float() @ self.W.float()
        # 1. compute the prediction by taking the argmax of the class scores
        # you might find torch.argmax useful here.
        label = np.argmax(scores.detach().numpy(),axis=1)
        # think about on what dimension (dim parameter) you should apply this operation
        return label

    def cross_entropy_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Convert class labels to one-hot encoding
        y_one_hot = torch.zeros(y.shape[0], self.num_classes)
        y_one_hot.scatter_(1, y.unsqueeze(1).long(), 1)

        # Calculate the cross-entropy loss manually
        loss = -torch.sum(y_one_hot * torch.log(y_pred))
        return loss

    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
       # use torch softmax function
       return torch.log(softmax(x))

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print('hyperparameters: lr {:.4f}, reg {:.4f}, epochs {:.2f}'.format(lr, reg_strength, epochs))
        for epoch in range(epochs):
            for ii in range((X_train.shape[0] - 1) // bs + 1):  # in batches of size bs
                start_idx = ii*bs  # we are ii batches in, each of size bs
                end_idx = (ii+1)*bs  # get bs examples

                # get the training training examples xb, and their coresponding annotations
                xb = X_train[start_idx:end_idx]
                yb = y_train[start_idx:end_idx]

                # apply the linear layer on the training examples from the current batch
                pred = self.predict_proba(xb)
                pred = self.log_softmax(pred)


                # compute the loss function
                # also add the L2 regularization loss (the sum of the squared weights)
                loss = self.cross_entropy_loss(pred, yb) + reg_strength * torch.sum(self.W ** 2)
                history.append(loss.detach().numpy())

                # start backpropagation: calculate the gradients with a backwards pass
                loss.backward()

                # update the parameters
                with torch.no_grad():  # we don't want to track gradients
                    # take a step in the negative direction of the gradient, the learning rate defines the step size
                    self.W -= self.W.grad * lr

                    # ATTENTION: you need to explictly set the gradients to 0 (let pytorch know that you are done with them).
                    self.W.grad.zero_()

        return history

    def get_weights(self, img_shape) -> np.ndarray:
        W = self.W.detach().numpy()
        # 0. ignore the bias term
        W = W[:-1,:]
        # 1. reshape the weights to (*image_shape, num_classes)
        W = W.reshape(img_shape[2],img_shape[0],img_shape[1],self.num_classes)
        # you might find the transpose function useful here
        W = np.transpose(W,(3,1,2,0))
        return W

    def load(self, path: str) -> bool:
        # load the input shape, the number of classes and the weight matrix from a file
        # you might find torch.load useful here
        self.W = torch.load(path)
        # don't forget to set the input_shape and num_classes fields
        self.num_classes = self.W.shape[1]
        self.input_shape = self.W.shape[0]
        return True

    def save(self, path: str) -> bool:
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find torch useful for this
        torch.save(self.W,path)
        return True

def configure_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=3e-4)

class Cifar10Classifier(nn.Module):
    def __init__(self):
            super(Cifar10Classifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def fit(self, X_train, y_train,epochs,bs):
        opt = configure_optimizer(self)
        loss_func = cross_entropy

        num_train = X_train.shape[0]

        epochs = tqdm.tqdm(range(epochs), desc="Epochs")

        for epoch in epochs:
            for ii in range((num_train - 1) // bs + 1):
                start_idx = ii * bs
                end_idx = start_idx + bs
                xb = X_train[start_idx:end_idx]
                yb = y_train[start_idx:end_idx]
                pred = self(xb) # call the forward function
                loss = loss_func(pred, yb.long()) # apply the loss function

                loss.backward() # start the backpropagation and compute the gradients
                opt.step() # apply the parameter update
                opt.zero_grad() # zero out the gradients

    def predict(self, X):
        import pdb; pdb.set_trace()
        return torch.argmax(self(X), dim=1)

        