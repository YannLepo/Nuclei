import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################################
### Network
class Net(nn.Module):
    def __init__(self, sizes):
        super(Net, self).__init__()
        
        sizes =  [20, 20, 20, 20]
        ks = 5
        pd = (ks-1)//2
        
        self.conv1 = nn.Conv2d(3, sizes[0], kernel_size=(ks,ks), padding=(pd,pd))
        self.conv1_1 = nn.Conv2d(sizes[0], sizes[0], kernel_size=(ks,ks), padding=(pd,pd))
        self.conv1_2 = nn.Conv2d(sizes[0], sizes[0], kernel_size=(ks,ks), padding=(pd,pd))
        self.conv2 = nn.Conv2d(sizes[0], sizes[1], kernel_size=(ks,ks), padding=(pd,pd))
        self.conv2_1 = nn.Conv2d(sizes[1], sizes[1], kernel_size=(ks,ks), padding=(pd,pd))
        self.conv3 = nn.Conv2d(sizes[1], sizes[2], kernel_size=(ks,ks), padding=(pd,pd))

        self.upsample3 = nn.ConvTranspose2d(sizes[2], sizes[1], 4, 2, 1)
        self.upsample2_1 = nn.ConvTranspose2d(sizes[1]+sizes[1], sizes[0], 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(sizes[1]+sizes[1], sizes[0], 4, 2, 1)
        self.upsample1 = nn.ConvTranspose2d(sizes[0]+sizes[0], sizes[3], 4, 2, 1)

        self.conv8 = nn.Conv2d(sizes[3]+sizes[0]+3, sizes[0], kernel_size=(3,3), padding=(1,1))
        self.conv9 = nn.Conv2d(sizes[0], 3, kernel_size=(1,1), padding=(0,0))
        
        self.epoch = 0
        
    def forward(self, x):

        y = []
        
        y.append( F.leaky_relu(self.conv1_1(F.leaky_relu(self.conv1(x), 0.1, inplace=True)), 0.1, inplace=True) )
        y.append( F.leaky_relu(F.max_pool2d(self.conv1_2(y[-1]), (2,2)), 0.1, inplace=True))
        y.append( F.leaky_relu(F.max_pool2d(self.conv2(y[-1]), (2,2)), 0.1, inplace=True))
        y.append( F.leaky_relu(F.max_pool2d(self.conv2_1(y[-1]), (2,2)), 0.1, inplace=True))
        y.append( F.leaky_relu(F.max_pool2d(self.conv3(y[-1]), (2,2)), 0.1, inplace=True))
        
        y.append( self.upsample3(y[-1]) )
        y.append( self.upsample2_1(torch.cat((y[-1], y[-3]), 1)) )
        y.append( self.upsample2(torch.cat((y[-1], y[-5]), 1)) )
        y.append( self.upsample1(torch.cat((y[-1], y[-7]), 1)) )
        y.append(self.conv8(torch.cat((y[-1], x, y[-9]), 1)))
        x =  self.conv9(y[-1])
                
        return x
