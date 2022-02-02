''' from computer vision A-Z course
    coded by trishit nath thakur'''


# Deep Convolutional GANs


# Importing the libraries


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Set hyperparameters


batchSize = 64 # We set the size of the batch.

imageSize = 64 # We set the size of the generated images


# Creating the transformations


transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 
 
 # Transforms are common image transformations. They can be chained together using Compose.We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

# Loading the dataset


dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

                    # shuffle=True implies internally the RandomSampler will be used, which just permutes the indices of all samples 
                    # num_workers tells data loader instance how many sub-processes to use for data loading

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.


def weights_init(m):
    classname = m.__class__.__name__           # The general rule for setting the weights in a neural network is to set them to be close to zero without being too small.
    
    if classname.find('Conv') != -1:    # if it has conv in its name as seen in ConvTranspose2d
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1: # if it has BatchNorm in its name as seen in BatchNorm2d
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator


class G(nn.Module):

    def __init__(self):
        
        super(G, self).__init__()
        
        self.main = nn.Sequential(                                # contain a sequence of modules (convolutions, full connections, etc.)
        
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),    
            
            # inversed convolution, 100 for input vector of size 100, 512 for number of feature maps of output, 4 kernel size,1 for stride, 0 for padding
            
            nn.BatchNorm2d(512), # normalising all features along dimension of batch(512)
            nn.ReLU(True),  # to break the linearity
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # similar to above
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),  # similar to above
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # similar to above
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),  # no of dimension for output is 3 corresponding to the 3 channels
            nn.Tanh()       # Tanh rectification to break the linearity and stay between -1 and +1
        )

    
    def forward(self, input):   # input that will be fed to the neural network, and that will return the output containing the generated images
        output = self.main(input)   # propagate the signal through the whole neural network of the generator defined by self.main
        return output

# Creating the generator


netG = G()

netG.apply(weights_init)    # initialize all the weights

# Defining the discriminator


class D(nn.Module):                  # inherit from the nn.Module tools

    def __init__(self):
        
        super(D, self).__init__()
        
        self.main = nn.Sequential(       # contain a sequence of modules (convolutions, full connections, etc
        
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),    # convolution , 3 channel for generated image of generator,64 for number of feature maps of output, 4 kernel size,2 for stride, 1 for padding
            nn.LeakyReLU(0.2, inplace = True),  # 0.2 for negetive slope and inplace true can do operation in place
            
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # similar to above
            nn.BatchNorm2d(128),                       # normalising all features along dimension of batch(128) 
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),  # similar to above
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),  # similar to above
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),   # 1 for output of discriminator which is simple vector of 1D containing a number bw 0 and 1 
            nn.Sigmoid()
        )

    
    def forward(self, input):    # input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1
        output = self.main(input)      # propagate the signal through the whole neural network of the discriminator defined by self.main
        return output.view(-1)      # view(-1) to flatten result so that all elements of output are along same dimension of batch size 

# Creating the discriminator


netD = D()

netD.apply(weights_init)

# Training the DCGANs


criterion = nn.BCELoss()
 
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))  

# optimizer object of the discriminator, betas is coefficient used for computing running averages of gradient and its square

optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))  

# optimizer object of the generator, betas is coefficient used for computing running averages of gradient and its square


for epoch in range(25):

    for i, data in enumerate(dataloader, 0):  # iterate over the images of the dataset
        
        # Updating the weights of the neural network of the discriminator

        netD.zero_grad()   # initialize to 0 the gradients of the discriminator with respect to the weights
        
        
        # Training the discriminator with a real image of the dataset
        
        real, _ = data    # getting a real image of the dataset which will be used to train the discriminator. 

        input = Variable(real)  # We wrap it in a variable(pytorch nn accepts input in form of torch variable ie it contains both a tensor and gradient)

        target = Variable(torch.ones(input.size()[0])) # 1 because real image dataset

        output = netD(input)

        errD_real = criterion(output, target)
        

        # Training the discriminator with a fake image generated by the generator
        
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # random input vector (noise) of the generator with (batch_size, number of elements, fake dimension for feature map
                                                                  # to get minibatch of size 100
        fake = netG(noise)

        target = Variable(torch.zeros(input.size()[0])) # 0 because fake image training

        output = netD(fake.detach())  # to save memory since fake is torch variable we remove gradient information since we wont need it

        errD_fake = criterion(output, target)
        

        # Backpropagating the total error
        
        errD = errD_real + errD_fake

        errD.backward()  # backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator

        optimizerD.step()  # apply the optimizer to update the weights


        # 2nd Step: Updating the weights of the neural network of the generator


        netG.zero_grad()    # initialize to 0 the gradients of the generator with respect to the weights

        target = Variable(torch.ones(input.size()[0]))

        output = netD(fake)

        errG = criterion(output, target)

        errG.backward()

        optimizerG.step()
        

        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps


        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
        if i % 100 == 0:         #  Every 100 steps:
            
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)   # save the real images of the minibatch

            fake = netG(noise)      # get our fake generated images

            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)  # save the fake generated images of the minibatch