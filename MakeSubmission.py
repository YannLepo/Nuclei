from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

### System related
from sys import path
from os import makedirs, listdir
from os.path import exists, isfile, join, isdir

### Useful functions for post processing and formatting for submission
from skimage.morphology import label # label regions
from scipy import misc, ndimage
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#################################################################################
### Read args from command line or take the default values
###TOBEDONE Clean args
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot-images', default='data/stage1_test/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--n_test', type=int, default=1000, help='number of test images')
parser.add_argument('--nc', type=int, default=3, help='number of color channels of input')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--plots'  , action='store_true', help='plot images')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net', default='default_dir/net.pth', help="path to net, if continued training")
parser.add_argument('--experiment', default='./experiments/Test5/', type=str, help='output directory')
parser.add_argument('--size_model', default='[10,10,10,10,10,10,10,10]', type=str, help='size of the model')

opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

kwargs = {'num_workers': 1, 'pin_memory': True} if opt.cuda else {}

#################################################################################
def saveImages(tensor, name):
    grid = vutils.make_grid(
        tensor, normalize=True, range=None, scale_each=False, nrow=9)
    vutils.save_image(grid, name)

#################################################################################
def list_directories(directory):
    list_of_dirs = [directory + f + '/images/'+ f + '.png'
                    for f in listdir(directory) if isdir(join(directory, f))]
    return sorted(list_of_dirs)

#################################################################################  
## Data class
class data():

    ## Initialization
    def __init__(self, path_images, nb_images = 10):
        
        ### Initialize
        self.path_images = path_images
        list_of_data = [ i for i in list_directories(self.path_images)]

        nb_images = min(len(list_of_data), nb_images)
        print(nb_images)

        ### Read images
        data = []
        for i in range(0, nb_images):
            print(list_of_data[0])
            ### Split string to get name
            name = list_of_data[0].split('/')[-1].split('.')[0]
            temp = torch.from_numpy(
                misc.imread(list_of_data[0], mode = 'RGB')).unsqueeze(0).transpose(0,-1).squeeze().float()

            data.append( [temp, name])
            list_of_data.remove(list_of_data[0])

        self.data = data
    #############################################################################       
    def __len__(self):
        return len(self.data)
    
    #############################################################################       
    def __getitem__(self, index):
        return self.data[ind][0], self.data[ind][1]
    
    #########################################################################
    def normalize(self):
        def norm(tensor):
            for i in range(0, tensor.size(0)):
                tensor[i].sub_(tensor[i].min())
                if tensor[i].max() - tensor[i].min() > 0:
                    tensor[i].mul_(1/(tensor[i].max() - tensor[i].min() ))
            return tensor
        
        self.data = [[norm(dt[0]), dt[1]] for dt in self.data ]

#################################################################################  
### Initialize dataset
dataset = data(opt.dataroot_images, nb_images = opt.n_test)
dataset.normalize()

################################################################################
### Read network
try:
    exists(opt.experiment) & exists('{0}/net.pth'.format(opt.experiment))
except ValueError:
    print("No trained model to read.")
    #TOBEDONE Stop execution

ngpu = int(opt.ngpu)
nc = int(opt.nc)

import net as net

opt.size_model = opt.size_model.replace('[','').replace(']','').split(',')
opt.size_model = [int(i) for i in opt.size_model]
model = net.Net(opt.size_model)

### If models exist, read their states and continue training, else initialize
print('Reading model {0}/net.pth'.format(opt.experiment) )
model.load_state_dict(torch.load('{:s}/net.pth'.format(opt.experiment)))
if opt.cuda:
    model.cuda()
model.eval()

opt.plots = True
################################################################################
### Compute loss on test set
for ind in range(0, len(dataset)):

    print("Process image ", ind)
    
    data, name = dataset[ind][0].unsqueeze(0), dataset[ind][1]
    if opt.cuda:
        data = dataset[ind].cuda()

    pad_bottom = 16-data.size(-2) % 16
    pad_right = 16-data.size(-1) % 16

    sizes=data.size()
    height = min(sizes[-2],sizes[-2]-sizes[-2] % 16)
    width = min(sizes[-1],sizes[-1]-sizes[-1] % 16)
    
    dt = data[:, :, :height, :width]
    if (sizes[-2]%16 != 0) | (sizes[-1]%16 != 0):
        dt = torch.cat((dt, data[:, :, sizes[-2] % 16:, sizes[-1] % 16:]), 0)
                 
    output = model(Variable(dt, volatile=True))

    classes = torch.Tensor(1, 1, sizes[-2], sizes[-1]).fill_(0.0)
    classes[:, :, :height, :width] = output.data[0:1].max(1)[1].unsqueeze(1)[:, :, :height, :width]
    if (sizes[-2]%16 != 0):
        classes[:, :, height:, -width:] = output.data[1:].max(1)[1].unsqueeze(1)[:, :, height-sizes[-2]%16:, :]
    if (sizes[-1]%16 != 0):
        classes[:, :, -height:, width:] = output.data[1:].max(1)[1].unsqueeze(1)[:, :, :, width-sizes[-1]%16:]
        
    if opt.plots:
        temp = torch.cat(
            (data.sum(1).unsqueeze(1), classes.float()), 1).view(-1, 1, data.size(-2), data.size(-1))
        saveImages(temp, '{:s}/Test{:s}.png'.format(opt.experiment, str(ind).zfill(3)))

#     mask_frontier = ((classes > 0) * (classes < 2)).squeeze().byte()
#     lab_img = torch.from_numpy( label( (classes > 1).squeeze().numpy() ) )
#     _, lab_ind = ndimage.distance_transform_edt((classes < 1).squeeze().numpy(), return_indices=True)

#     ### Attribute frontiers to closest label
#     lab_ind_height = torch.from_numpy(lab_ind[0])[mask_frontier].long().view(-1)
#     lab_ind_width = torch.from_numpy(lab_ind[1])[mask_frontier].long().view(-1)
#     lab_img[mask_frontier] = lab_img[lab_ind_height,:][:, lab_ind_width]

#     ### If labeled regions are non connexe divide into two labels
#     ### TOBETESTED
#     lab_max = lab_img.max() + 1
#     for i in range(1, lab_img.max()+1):
#         label_of_label =  label( (lab_img==i).numpy() )
#         ###
#         lab_img[ lab_img==i & torch.from_numpy(label_of_label) > 1 ] += lab_max
#         lab_max += len( np.unique( label_of_label )) - 2
#         ###
#         for j in range(2, len( np.unique( label_of_label )) ):
#             lab_img[ lab_img==i & torch.from_numpy(label_of_label) == j ] = lab_max
#             lab_max += 1

#     ### TOBEDONE Check if filling is needed
    
#     if lab_img.max()<1:
#         lab_img[0,0] = 1 # ensure at least one prediction per image
    
#     out_pred_list = []
#     for i in range(1, lab_img.max()+1):
       
#         dots = np.where( (lab_img==i).numpy().T.flatten()==1)[0] # .T sets Fortran order down-then-right

#         run_lengths = []
#         prev = -2
#         for b in dots:
#             if (b>prev+1): run_lengths.extend((b+1, 0))
#             run_lengths[-1] += 1
#             prev = b
            
#         print(name, run_lengths)
#     out_pred_list+=[dict(ImageId=name, EncodedPixels = ' '.join(np.array(run_lengths).astype(str)))]

# out_pred_df = pd.DataFrame(out_pred_list)
