from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils

### System related
from sys import path
from os import makedirs
from os.path import exists

### Plots
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

### Generic routines
import functions

#################################################################################
### Read args from command line or take the default values
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot-images', default='../data', help='path to dataset')
parser.add_argument('--dataroot-masks', default='../data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the size of the input image to network')
parser.add_argument('--n_train', type=int, default=1, help='number of train images')
parser.add_argument('--n_test', type=int, default=1, help='number of test images')
parser.add_argument('--n_crops', type=int, default=10, help='number of patches per images')
parser.add_argument('--nc', type=int, default=3, help='number of color channels of input')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--plots'  , action='store_true', help='plot images')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='default_dir/ssd.pth', help="path to net, if continued training")
parser.add_argument('--experiment', default='./Test', type=str, help='output directory')
parser.add_argument('--optim', default='adam', help='[adam]|sgd|rms|adadelta')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--size_model', default='[10,10,10,10,10,10,10,10]', type=str, help='size of the model')
parser.add_argument('--thres', type=float, default=0.66, help='threshold for classification')

opt = parser.parse_args()
opt.manualSeed = opt.seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
CUDA_LAUNCH_BLOCKING=1
cudnn.benchmark = False

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}

#################################################################################
def saveImages(tensor, name, directory='./Images/'):
    grid = vutils.make_grid(tensor, normalize=True, range=None, scale_each=False, nrow=9)
    vutils.save_image(grid, directory+name)

#################################################################################    
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

from skimage.morphology import label # label regions
def mask_to_rles(mask):
    lab_img = label(mask.squeeze().numpy() > 0.5)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)

#################################################################################
##### Functions        
############################
### Train function 
def train(model, optimizer, epoch):
    model.train() 
    cum_loss = 0

    loss = Variable(torch.Tensor([0.0]))
    
    for data, target in dataloader:

        if opt.cuda:
            data, loss, target = data.cuda(), loss.cuda(), target.cuda()
 
        ### Optimize network for its task
        model.zero_grad()
        output = model(Variable(data))
        loss = custom_loss(output, target)

        loss.backward()
        optimizer.step()

        cum_loss += loss.data[0]
        
    ### Print and save
    cum_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(cum_loss))

    ### Save models states
    model.epoch = model.epoch + 1
    torch.save(model.state_dict(), '{:s}/net.pth'.format(opt.experiment))
    
    return cum_loss

############################
# Test function
def test(model, dtloader, epoch):
    model.eval()
    test_loss = 0

    ### Compute loss on test set
    for ind, sample in enumerate(dtloader):
        if opt.cuda:
            data, target = sample[0].cuda(), sample[1].cuda()
            
        output = model(Variable(data, volatile=True))
        thres = (output.data > 0.66) + (output.data > 0.40) 
        #thres = output.data.max(1)[1].unsqueeze(1)
        target_thres = (target > 0.66) + (target > 0.40) 
        
        if opt.plots and ind==0 :
            temp = torch.cat(
                (data.sum(1).unsqueeze(1), target_thres.float(), thres.float()), 1).view(-1, 1, data.size(-2), data.size(-1))
            saveImages(temp, "Test{:s}.png".format(str(epoch).zfill(3)), directory="./Images/")

        ### TOBETESTED 
        ### train_row_rles = list(mask_to_rles(mask))
            
        test_loss += custom_loss(output, target, vol=True).data[0]
            
    test_loss /= len(dtloader) # loss function already averages over batch size
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss

#################################################################################
########     Main code    #######################################################
#################################################################################
###Load the dataset

opt.nc = 1
import dataset as nuclei

dataset = nuclei.data(opt.dataroot_images, opt.dataroot_masks, image_size = opt.imageSize,
                      nb_images = opt.n_train, nb_crops = opt.n_crops)
dataset.normalize()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,  **kwargs)

testset = nuclei.data(opt.dataroot_images, opt.dataroot_masks, list_of_data = dataset.remaining_data,
                      nb_images = opt.n_train, nb_crops = 1, image_size = opt.imageSize,)
testset.normalize()
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=True,  **kwargs)

################################################################################
### Create/Read network and initialize

if not exists(opt.experiment):
    makedirs(opt.experiment)

ngpu = int(opt.ngpu)
nc = int(opt.nc)

import models.net as net

opt.size_model = opt.size_model.replace('[','').replace(']','').split(',')
opt.size_model = [int(i) for i in opt.size_model]

model = net.Net(opt.size_model)

### If models exist, read their states and continue training, else initialize
if exists('{0}/net.pth'.format(opt.experiment)):
    print('Reading model {0}/net.pth'.format(opt.experiment) )
    model.load_state_dict(torch.load('{:s}/net.pth'.format(opt.experiment)))
else:
    model.apply(functions.weights_init)

if opt.cuda:
    model.cuda()
    
### Setup optimizer
if opt.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif  opt.optim == 'rms':
    optimizer = optim.RMSprop(model.parameters(), lr = opt.lr, momentum=0)
elif  opt.optim == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr= opt.lr, rho=0.9,
                               eps=1e-06, weight_decay=0)
elif  opt.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    
assert optimizer

### Additional variables
log = {'train_acc', 'test_acc'}

train_error = []
test_error = []

#################################################################################
### Print options and models arch
print(opt)
print(model)

#################################################################################
# Loss

# weight = torch.Tensor([1, 1]) # Weight applied on classes while computing the loss
# if opt.cuda:
#    weight = weight.cuda()
# criterion = nn.NLLLoss2d()
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

def custom_loss(output, target, vol=False):
    return criterion(output, Variable(target, volatile=vol))

################################################################################
### Train

last_reduction = 0
for epoch in range(1, opt.epochs+1):
    print("Epoch {:d} ------------------------------------------".format(epoch))
    
    train_error.append( train(model, optimizer, epoch) )
    test_error.append( test(model, testloader, epoch) )
    
    if len(train_error) >= 45 and epoch >= last_reduction + 10:
        if functions.reduce_learning_rate(optimizer, train_error, opt):
            last_reduction = epoch # to keep learning rate for at least 10 epochs




            

