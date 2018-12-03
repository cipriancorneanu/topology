'''
Utilities  for saving data to disk
'''


import os
import numpy as np
import torch
import pickle as pkl 
import h5py 


''' Define a function that saves model and activations at a set of epochs '''
def save():
    soemthing = 0

    
def save_checkpoint(checkpoint, path, fname):
    """ Save checkpoint to path with fname """

    print('Saving checkpoint...')
    
    if not os.path.isdir(path):
        os.mkdir(path)
        
    torch.save(checkpoint, path+fname)

        
def save_activations(activs, targets, path, fname, internal_path):
    """
    Save activs and targets to path/fname in h5py dataset.
    Activs saved at <internal_path>/activations/layer_<layer>/.
    Targets saved at <internal_path>/targets/.
    """
    
    print('Saving activations...')

    ''' Create file '''
    if not os.path.exists(path):
        os.makedirs(path)
    file = h5py.File(path+fname, 'a')
    
    ''' Save activations; if exists replace '''
    for i, x in enumerate(activs):
        dts = internal_path+"/activations/layer_"+str(i)
        if dts in file:
            data = file[dts]
            data[...] = x
        else:
            file.create_dataset(internal_path+"/activations/layer_"+str(i), data=x, dtype=np.float16)

    ''' Save targets; if exists replace '''
    dts = internal_path+"/targets"
    if dts in file:
        data =  file[dts]
        data[...] = x
    else:
        file.create_dataset(internal_path+"/targets", data=targets)

    file.close()

    
def save_losses(losses, path, fname):
    """ Save losses (np.ndarray) to path with fname """
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(path+fname, 'wb') as f:
        pkl.dump(losses, f, protocol=2)
