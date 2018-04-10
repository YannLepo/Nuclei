from os import listdir
from os.path import isfile, join, isdir
from random import seed, randint, uniform

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
                
                temp_mask = torch.Tensor(3, temp.size(-2), temp.size(-1)).fill_(0.0)
                temp_mask[0] = 1.0
                
                test = None
                for i in range(len(list_of_data[0][1])):
                    # ### Binary image
                    # temp_mask += torch.from_numpy(
                    #     misc.imread(list_of_data[0][1][i], mode = 'F'))

                    # ### Labeled image: Background = 0, boundary = 1, cell = 2
                    # tmp =  misc.imread(list_of_data[0][1][i], mode = 'F')
                    # er = ndimage.binary_erosion(tmp).astype(uint8)
                    # eroded = (torch.from_numpy( ndimage.binary_erosion( er ).astype(uint8) ) ).float()
                    # # eroded = (torch.from_numpy( ndimage.binary_erosion( tmp ).astype(uint8) ) ).float()
                    # temp_mask += 2 * eroded  + ((torch.from_numpy(tmp)>0).float() - eroded)

                    ### Hot vector : Background = 0, boundary = 1, cell = 2
                    tmp =  misc.imread(list_of_data[0][1][i], mode = 'F')
                    er = ndimage.binary_erosion(tmp).astype(uint8)
                    eroded = (torch.from_numpy( ndimage.binary_erosion( er ).astype(uint8) ) )

                    temp_mask[0][ eroded  + ((torch.from_numpy(tmp)>0) - eroded)] = 0.0
                    temp_mask[1][ ((torch.from_numpy(tmp)>0) - eroded) ] = 1.0
                    temp_mask[2][ eroded ] = 1.0
                                             
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
        def flipTensor(img):
            width = img[0].size(-1)
            height = img[0].size(-2)
            if randint(0,1):
                img[0] = img[0].index_select(-1, torch.linspace(width-1, 0, width).long())
                img[1] = img[1].index_select(-1, torch.linspace(width-1, 0, width).long())
            if randint(0,1):
                img[0] = img[0].index_select(-2, torch.linspace(height-1, 0, height).long())
                img[1] = img[1].index_select(-2, torch.linspace(height-1, 0, height).long())
            return img

        def getImage(ind):
            height = self.data[ind][0].size(-2)
            width = self.data[ind][0].size(-1)
            image_size = self.image_size
              
            rand_h = randint(0, max(0, height-image_size))
            rand_w = randint(0, max(0, width-image_size)) 

            image = self.data[ind][0][:,rand_h:rand_h+image_size,rand_w:rand_w+image_size]
            mask =  self.data[ind][1][:,rand_h:rand_h+image_size,rand_w:rand_w+image_size]

            return flipTensor([image, mask])
         
        ### Get an image and its mask, crop everything to a fixed size
        cur_ind = index // self.nb_crops
        rand_ind = randint(0, len(self.data)-1)

        tmp_ind = getImage(cur_ind)
        tmp_randind = getImage(rand_ind)

        lamb = uniform(0, 0.3)
        
        image = lamb * tmp_ind[0] + (1-lamb) * tmp_randind[0]
        mask = lamb * tmp_ind[1].float() + (1-lamb) * tmp_randind[1].float()
        
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
        
        self.data = [[norm(dt[0]), dt[1].long()] for dt in self.data ]

#################################################################################
