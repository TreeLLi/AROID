import torch as tc
from torchvision.transforms import functional as F

class Cutout(tc.nn.Module):
    def __init__(self, strength):
        super().__init__()

        self.strength = strength
        
    def forward(self, img):
        strength = self.strength
        width, height = F.get_image_size(img)
        
        radius = strength // 2
        
        x_min, x_max = -radius, width-radius
        y_min, y_max = -radius, height-radius

        if x_min == x_max:
            x_top = x_min
        else:
            x_top = tc.randint(x_min, x_max, (1,)).item()

        if y_min == y_max:
            y_top = y_min
        else:
            y_top = tc.randint(y_min, y_max, (1,)).item()

        x_bot = x_top + strength
        y_bot = y_top + strength
        x_top = max(0, x_top)
        y_top = max(0, y_top)
        x_bot = min(x_bot, width)
        y_bot = min(y_bot, height)
        
        width = x_bot - x_top
        height = y_bot - y_top

        return F.erase(img, y_top, x_top, height, width, 0)
