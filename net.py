import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

from torch.autograd import Variable

# #################################################################################
# ### Network
# class Net(nn.Module):
#     def __init__(self, sizes):
#         super(Net, self).__init__()
        
#         sizes =  [10, 10, 10, 10, 10, 10, 10, 10]
#         self.conv1 = nn.Conv2d(1, sizes[0], kernel_size=(3,3), padding=(1,1))
#         self.conv2 = nn.Conv2d(sizes[0], sizes[1], kernel_size=(3,3), padding=(1,1))
#         self.conv3 = nn.Conv2d(sizes[1], sizes[2], kernel_size=(3,3), padding=(1,1))
#         self.conv3_1 = nn.Conv2d(sizes[2], sizes[3], kernel_size=(3,3), padding=(1,1))
#         self.conv4 = nn.Conv2d(sizes[3], sizes[4], kernel_size=(3,3), padding=(1,1))

#         self.deconv4 = nn.ConvTranspose2d(
#             sizes[4], sizes[5], kernel_size=1, stride=1, padding=0)
#         self.deconv3 = nn.ConvTranspose2d(
#             sizes[5]+sizes[3]+2, sizes[6], kernel_size=3, stride=1, padding=1, output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(
#             sizes[6]+sizes[1]+2, sizes[7], kernel_size=3, stride=1, padding=1, output_padding=1)
        
#         self.predict4 = nn.Conv2d(sizes[5]+sizes[4], 1, kernel_size=3, stride=1, padding=1)
#         self.predict3 = nn.Conv2d(sizes[5]+sizes[3]+1, 1, kernel_size=3, stride=1, padding=1)
#         self.upsample3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.predict2 = nn.Conv2d(sizes[6]+sizes[1]+1, 1, kernel_size=3, stride=1, padding=1)
#         self.upsample2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        
#         self.predict1 = nn.ConvTranspose2d(
#             sizes[7]+sizes[0]+2, 2, kernel_size=5, stride=2, padding=2, output_padding=1)

#         self.epoch = 0
                                          
        
#     def forward(self, x):
#         x1 = F.leaky_relu(F.max_pool2d(self.conv1(x), (2,2)), 0.1, inplace=True)
#         x2 = F.leaky_relu(F.max_pool2d(self.conv2(x1), (2,2)), 0.1, inplace=True)
#         x = F.leaky_relu(self.conv3(x2))
#         x3 = F.leaky_relu(F.max_pool2d(self.conv3_1(x), (2,2)), 0.1, inplace=True)
#         x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        
#         x = F.leaky_relu(self.deconv4(x4), 0.2, inplace=True)
#         xf3 = self.predict4(torch.cat((x, x4),1))
        
#         xf2 = self.upsample3(self.predict3(torch.cat((x, x3, xf3), 1)))
#         x = F.leaky_relu(self.deconv3(torch.cat((x, x3, xf3), 1)), 0.1, inplace=True)

#         xf1 = self.upsample2(self.predict2(torch.cat((x, x2, xf2), 1)))
#         x = F.leaky_relu(self.deconv2(torch.cat((x, x2, xf2), 1)), 0.2, inplace=True)

#         x = self.predict1(torch.cat((x, x1, xf1), 1))
        
#         return x #(x, xf1, xf2, xf3)



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
        self.conv3 = nn.Conv2d(sizes[1], sizes[2], kernel_size=(ks,ks), padding=(pd,pd))

        self.upsample3 = nn.ConvTranspose2d(sizes[2], sizes[1], 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(sizes[1]+sizes[1], sizes[0], 4, 2, 1)
        self.upsample1 = nn.ConvTranspose2d(sizes[0]+sizes[0], sizes[3], 4, 2, 1)
        
        self.conv8 = nn.Conv2d(sizes[3]+sizes[0]+3, 3, kernel_size=(1,1), padding=(0,0))
        
        self.epoch = 0
        
    def forward(self, x):
        x0 = F.leaky_relu(self.conv1_1(F.leaky_relu(self.conv1(x), 0.1, inplace=True)), 0.1, inplace=True)
        x1 = F.leaky_relu(F.max_pool2d(self.conv1_2(x0), (2,2)), 0.1, inplace=True)
        x2 = F.leaky_relu(F.max_pool2d(self.conv2(x1), (2,2)), 0.1, inplace=True)
        x3 = F.leaky_relu(F.max_pool2d(self.conv3(x2), (2,2)), 0.1, inplace=True)
        
        x4 = self.upsample3(x3)
        x5 = self.upsample2(torch.cat((x4, x2), 1))
        x6 = self.upsample1(torch.cat((x5, x1), 1))
        x = F.log_softmax(self.conv8(torch.cat((x6, x, x0), 1)))
                
        return x
