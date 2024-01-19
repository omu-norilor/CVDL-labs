import os
import sys
import torch
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from activations import softmax,sigmoid
import cifar10
from model import SoftmaxClassifier, Cifar10Classifier
from metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch import optim


def activations_validation():
    # validate softmax
    # let's check that you obtained the same values
    # as the softmax implementation in torch
    arr = np.array([2, 4, 10, 100, 2.0])
    torch_softmax = torch.nn.functional.softmax(torch.from_numpy(arr), dim=0).numpy()
    custom_softmax = softmax(arr) # Ensure the data type is the same
    assert np.allclose(torch_softmax, custom_softmax)

    arr = np.array([0.0, 0, 0, 1, 0])
    torch_softmax = torch.nn.functional.softmax(torch.from_numpy(arr), dim=0).numpy()
    custom_softmax = softmax(arr).astype(np.float32)  # Ensure the data type is the same

    assert np.allclose(torch_softmax, custom_softmax)


def activations_plot():
    x= np.asarray([20, 30, -15, 45, 39, -10])
    # x =np.linspace(-10,10,100)
    T = [0.25, 0.75, 1, 1.5, 2, 5, 10, 20, 30]

    for idx in range(0, len(T)):
    # plot the result of applying the softmax function
        x_softmaxed = softmax(x=x,t=T[idx])
        plt.title(f"normal vs softmax with t {T[idx]}")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.bar(x,x_softmaxed,color="green")
        plt.show()
    # with different temperatures on the array x


def model_init_test():
    cifar_root_dir = 'cifar-10-batches-py'
    _, _, X_test, y_test = cifar10.load_ciaf10(cifar_root_dir)
    indices = np.random.choice(len(X_test), 15)

    # show random images from indices
    display_images, display_labels = X_test[indices], y_test[indices]
    for idx, (img, label) in enumerate(zip(display_images, display_labels)):
        plt.subplot(3, 5, idx + 1)
        # here i reshaped the image to (3, 32, 32) and then transposed it to (32, 32, 3) 
        # cuz of the numpy convention and imshow convention (idiotic, i know)
        # later i saw this transformation was to be implemented in the load_cifar10 function, so i did it there
        # i still left the code here, just for the sake of it, i was glad i figured it out on my own
        # img_reshaped = img.reshape((3, 32, 32)).transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(cifar10.LABELS[label])
        plt.tight_layout()

    plt.show()
    num_pixels = np.prod(X_test[0].shape)
    cls = SoftmaxClassifier(num_pixels+1, len(cifar10.LABELS))

    test_example = torch.from_numpy(np.append(X_test[0].flatten(), 1.0)).float()
    test_example = test_example[None, :]

    # import pdb; pdb.set_trace()
    print('predicted class ', cifar10.LABELS[cls.predict(test_example).tolist()[0]], cls.predict(test_example))
    print('probas: ', cls.predict_proba(test_example).detach().numpy())


def model_flow():
    cifar_root_dir = 'cifar-10-batches-py'

    # load cifar10 dataset
    X_train, y_train, X_test, y_test = cifar10.load_ciaf10(cifar_root_dir)

    # convert the training and test data to floating point
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Reshape the training data such that we have one image per row
    print(f"Before reshaping {X_train.shape}")
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(f"After reshaping {X_train.shape}")

    # pre-processing: subtract mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # Bias trick - add 1 to each training example
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    # convert everything to tensors
    X_train, y_train, X_test, y_test = map(
        torch.tensor, (X_train, y_train, X_test, y_test)
    )

    X_train = X_train.float()
    X_test = X_test.float()


    if not os.path.exists('train'):
        os.mkdir('train')

    best_acc = -1
    best_cls_path = ''


    input_size_flattened = reduce((lambda a, b: a * b), X_train[0].shape)

    # the batch size
    batch_size = 200
    # number of training steps per training process
    train_epochs = 50


    lr = 0.007 # change the value - hyperparameter tuning
    reg_strength = 0.007 # change the value - hyperparameter tuning

    cls = SoftmaxClassifier(input_shape=input_size_flattened, num_classes=cifar10.NUM_CLASSES)
    history = cls.fit(X_train, y_train, lr=lr, reg_strength=reg_strength,
            epochs=train_epochs, bs=batch_size)

    with torch.no_grad():
        y_train_pred = cls.predict(X_train)
        y_val_pred = cls.predict(X_test)

    train_acc = torch.mean((torch.Tensor(y_train) == torch.Tensor(y_train_pred)).float())


    test_acc = torch.mean((torch.Tensor(y_test) == torch.Tensor(y_val_pred)).float())
    sys.stdout.write('\rlr {:.4f}, reg_strength{:.2f}, test_acc {:.2f}; train_acc {:.2f}'.format(lr, reg_strength, test_acc, train_acc))
    cls_path = os.path.join('train', 'softmax_lr{:.4f}_reg{:.4f}-test{:.2f}.npy'.format(lr, reg_strength, test_acc))
    cls.save(cls_path)


    plt.plot(history)
    plt.show()

    best_softmax = cls


    plt.rcParams['image.cmap'] = 'gray'
    # now let's display the weights for the best model
    weights = best_softmax.get_weights((32, 32, 3))

    w_min = np.amin(weights)
    w_max = np.amax(weights)

    for idx in range(0, cifar10.NUM_CLASSES):
        plt.subplot(2, 5, idx + 1)
        # normalize the weights
        template = 255.0 * (weights[idx, :, :, :].squeeze() - w_min) / (w_max - w_min)
        template = template.astype(np.uint8)
        plt.imshow(template)
        plt.title(cifar10.LABELS[idx])

    plt.show()

    # use the metrics module to compute the precision, recall and confusion matrix for the best classifier
    conf_mat = confusion_matrix(y_test, y_val_pred, num_classes=cifar10.NUM_CLASSES)
    precision = precision_score(y_test, y_val_pred, num_classes=cifar10.NUM_CLASSES)
    recall = recall_score(y_test, y_val_pred, num_classes=cifar10.NUM_CLASSES)
    accuracy = accuracy_score(y_test, y_val_pred) 

    if(best_acc < accuracy):
        best_acc = accuracy
        best_cls_path = cls_path
    
    # save the best classifier
    best_softmax.save(best_cls_path)
    print("Confusion matrix: ")
    print(conf_mat)
    print("Precision: ")
    print(precision)
    print("Recall: ")
    print(recall)


def fuck_it_lets_pytorch():
    model = Cifar10Classifier()
    print(*list(model.children()))
    print(*list(model.parameters()), sep="\n")
    loss_func=cross_entropy
    cifar_root_dir = 'cifar-10-batches-py'
    # load cifar10 dataset
    X_train, y_train, X_test, y_test = cifar10.load_ciaf10(cifar_root_dir)

    # convert the training and test data to floating point
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Reshape the training data such that we have one image per row
    # import pdb;pdb.set_trace()
    X_train = X_train.transpose(0,3,1,2)
    X_test = X_test.transpose(0,3,1,2)
    

    # pre-processing: subtract mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # convert everything to tensors
    X_train, y_train, X_test, y_test = map(
        torch.tensor, (X_train, y_train, X_test, y_test)
    )

    X_train = X_train.float()
    X_test = X_test.float()

    print("before training:", loss_func(model(X_test), y_test.long()), sep="\n\t")
    epochs = 50
    bs = 64
    model.fit(X_train, y_train, epochs, bs)

    y_train_pred = model.predict(X_train)
    train_acc = f1_score(y_train, y_train_pred)
    y_val_pred = model.predict(X_test)
    test_acc = f1_score(y_test, y_val_pred)
    print('Train acc ', train_acc, ' test acc ', test_acc)
    print("after training:", loss_func(model(X_test), y_test.long()), sep="\n\t")


if __name__ == "__main__":
    # activations_validation()
    # activations_plot()
    # model_init_test()
    # model_flow()
    fuck_it_lets_pytorch()