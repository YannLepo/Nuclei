from os import listdir
from os.path import isfile, join, isdir
from random import seed, randint, random
seed(1)

import torch
import torchvision
from numpy import sqrt, uint8

from scipy import misc, ndimage

#################################################################################
## List files contained in directory
def list_files(directory):
    list_of_files = [directory +'/'+ f
                     for f in listdir(directory) if isfile(join(directory, f))]
    return sorted(list_of_files)

def list_directories(directory):
    list_of_dirs = [directory + '/'+ f
                    for f in listdir(directory) if isdir(join(directory, f))]
    return sorted(list_of_dirs)

#################################################################################  
## Dataset class
class data():

    ## Initialization
    def __init__(self, path_images, path_masks, list_of_data = None, image_size = 64,
                 nb_images = 10, nb_crops = 10):
        
        ### Initialize
        self.path_images = path_images
        self.path_masks = path_masks
        
        if list_of_data is None:
            images = [ i for i in list_files(self.path_images)]
            masks = [ list_files(i) for i in list_directories(self.path_masks)]
            
            assert len(images) == len(masks)
            list_of_data = [[images[i], masks[i]] for i in range(len(images))]

        nb_images = min(len(list_of_data), nb_images)
        print(nb_images * nb_crops)
        
        ### Read images
        data = []
        for i in range(0, nb_images):
            if len(list_of_data) > 0:
                temp = torch.from_numpy(
                    misc.imread(list_of_data[0][0], mode = 'RGB')).unsqueeze(0).transpose(0,-1).squeeze().float()
                
                temp_mask = torch.Tensor(1, temp.size(-2), temp.size(-1)).fill_(0.0)
                
                for i in range(len(list_of_data[0][1])):
                    tmp =  misc.imread(list_of_data[0][1][i], mode = 'F')
                    eroded = (torch.from_numpy( ndimage.binary_erosion(tmp).astype(uint8)) ).float()
                    temp_mask += eroded + 0.5 * (torch.from_numpy(tmp).float() - eroded) 
                    
                    # temp_mask += torch.from_numpy(
                    #     misc.imread(list_of_data[0][1][i], mode = 'F'))

                data.append( [temp, temp_mask] )
                list_of_data.remove(list_of_data[0])
                
            else:
                print("Not enough images in the directory")
                print("Only", len(self.names), "images read")
                break

        self.remaining_data = list_of_data
        self.data = data
        self.nb_crops = nb_crops
        self.image_size = image_size
            
    #############################################################################       
    def __getitem__(self, index):


        ### Random flip
        def flipTensor(img, dim):
            return img.index_select(dim, torch.linspace(img.size(dim)-1, 0, img.size(dim)).long())
        
        ### Get an image and its mask, crop everything to a fixed size
        ind = index // self.nb_crops

        height = self.data[ind][0].size(-2)
        width = self.data[ind][0].size(-1)
        image_size = self.image_size
              
        rand_h = randint(0, max(0, height-image_size))
        rand_w = randint(0, max(0, width-image_size)) 

        image = self.data[ind][0][:,rand_h:rand_h+image_size,rand_w:rand_w+image_size]
        mask =  self.data[ind][1][:,rand_h:rand_h+image_size,rand_w:rand_w+image_size]

        # flip horizontally
        if randint(0,1):
            image = flipTensor(image, -1)
            mask = flipTensor(mask, -1)
            
        # flip vertically
        if randint(0,1):
            image = flipTensor(image, -2)
            mask = flipTensor(mask, -2)
        
        return image, mask
    
    #############################################################################
    def __len__(self):
        return len(self.data) * self.nb_crops
        
    #########################################################################
    def normalize(self):
        def norm(tensor):
            for i in range(0, tensor.size(0)):
                tensor[i].sub_(tensor[i].min())
                if tensor[i].max() - tensor[i].min() > 0:
                    tensor[i].mul_(1/(tensor[i].max() - tensor[i].min() ))
            return tensor
        
        self.data = [[norm(dt[0]), norm(dt[1])] for dt in self.data ]

#################################################################################
