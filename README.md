# CVDL-labs
 
#-------------------------------------------------#

LAB1 - Basics

Smooth sail here, nothing complicated.

#-------------------------------------------------#

LAB2 - Softmax Classifier

Low performance models. Got used to numpy at least.

#-------------------------------------------------#

LAB3 - Convolutions

Had some problems with zero padding and filters.
Transfer training worked nicely.

#-------------------------------------------------#

LAB4 - Dataset and Dataloader

Had to juggle with the pictures.
What i did:
- got the common names from the input and output
folders
- got the frequencies for each common names
- only used the the people that appear in both 
folders the same ammount of times

#-------------------------------------------------#

LAB5 - Unet trial and error

Building an Unet from scratch was interesting.
I got very near to the correct architecture and 
then gave up and looked online, just to see that 
I was 3 steps away from succes.

Had some problems with training speed due to batch 
size. It took me some time to find the problem.

Wandb is nice, its similar to tensorboard but also 
provides the hyperparameter optimisation that i did 
not use.

I am still having problems with training stability,
even though i implemented learning decay and early 
stopping to prevent the model's instability.

I would like to find out how to fix those problems.

#UPDATE# dice loss was faulty, now it works great!

#-------------------------------------------------#

LAB6 - Production style ML 

Gradio is a bit of pain. It wants enumerations,
fixed indices and so on. I need to modify the Unet 
quite a lot to achieve the deploy phase sadly.

I feel like flask/torchserve + some random php may
be easier to do.

I won't do the deploy, but i once deployed a ML
model on AWS lambda. Took me some days to navigate
the aws dense forest and find my path.

#-------------------------------------------------#

LAB NOTES 
- I really liked the Unet
- I hated it when I got awful performance or bugs 
after working a lot on models and getting to feel 
confident on my work 

#-------------------------------------------------#
