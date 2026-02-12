'''
This script will load the original HMR model with Torch 2.4.0 and save it in a backward compatible way so that it can be loaded with older versions of PyTorch, eg. 2.1.0.
Run this script with Torch 2.4.0, and then try loading the saved model with older PyTorch versions to verify backward compatibility. 
'''

import torch

m = torch.jit.load("vith/sam-3d-body-vith-mhr_model.pt", map_location="cpu")

m = torch.jit.freeze(m)
m = torch.jit.optimize_for_inference(m)

torch.jit.save(m, "vith/mhr_compat_try1.pt")