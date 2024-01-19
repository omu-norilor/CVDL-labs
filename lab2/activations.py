import numpy as np
import torch
def softmax(x, t=1):
    """
    Applies the softmax temperature on the input x, using the temperature t
    """
    e_x = torch.exp(x / t)
    result = (e_x / e_x.sum()).float()
    return result

def sigmoid(x, t=1):
    return 1/(1+np.exp(x/t))

if __name__== "__main__":
    print(softmax(np.array([1, 2, 3, 4, 1, 2, 3]), t=1))