import math
import torch as tc
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode as Interpolation

from src.data.idbh import CropShift
from src.data.policy import COLSHA_SPACE_M, erase, colorshape

BIN_N = 11

class UniformAROID(tc.nn.Module):
    operations = list(COLSHA_SPACE_M.keys())

    def __init__(self):
        super().__init__()

        self.flip = T.RandomHorizontalFlip()
        self.crop = CropShift(0, 16)
                
    def forward(self, img):
        img = self.flip(img)
        img = self.crop(img)
        
        n_ops = len(self.operations)
        
        roll = tc.randint(0, n_ops, (1,)).item()
        op = self.operations[roll]

        strengths = COLSHA_SPACE_M[op]
        n_strengths = len(strengths)
        if n_strengths > 0:
            roll = tc.randint(0, n_strengths, (1,)).item()
            strength = strengths[roll]
        else:
            strength = None

        img = colorshape(img, op, strength)
        img = F.to_tensor(img)

        roll = tc.randint(0, BIN_N, (1,)).item()
        if roll != 0:
            img = erase(img, roll, BIN_N)
        
        return img

