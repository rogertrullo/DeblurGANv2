import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from skimage.measure import compare_ssim as SSIM

from util.metrics import PSNR


class Encoder(nn.Module):
    def __init__(self, inchans=1, nz=32):
        super(Encoder, self).__init__()
        self.convlist=nn.ModuleList([nn.Conv2d(inchans,32, kernel_size=4, stride=2, padding=1),#x/2
                                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),#x/4 /2
                                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),#x/4 /1
                                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),#x/8 /2
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),#x/8 /1
                                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),#x/16 /2
                                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),#x/16 /1
                                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),#x/16 /1
                                    nn.Conv2d(32, nz, kernel_size=8, stride=1, padding=0)])#x/(16*8) /8
    def forward(self, x):
        for i, l in enumerate(self.convlist[:-1]):
            x=l(x)
            x=F.leaky_relu(x, negative_slope=0.2)
        
        out=self.convlist[-1](x)
        return out
class Decoder(nn.Module):
    def __init__(self, outchans=1, nz=32):
        super(Decoder, self).__init__()
        self.convlist=nn.ModuleList([
            nn.ConvTranspose2d(nz,32,kernel_size=8, stride=8, padding=0), #x8   
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#x1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),#x1
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),#x2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),#x1
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),#x2
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),#x1
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),#x2
            nn.ConvTranspose2d(32,outchans, kernel_size=4, stride=2, padding=1)#x2
        ])
    def forward(self, x):
        for i, l in enumerate(self.convlist[:-1]):
            x=l(x)
            x=F.leaky_relu(x, negative_slope=0.2)
        
        out=self.convlist[-1](x)
        return out
class CAE(nn.Module):
    def __init__(self, inchans=3, nz=32):
        super(CAE, self).__init__()
        self.enc=Encoder(inchans=inchans, nz=nz)
        self.dec=Decoder(outchans=inchans, nz=nz)
    def forward(self, x):
        z=self.enc(x)
        out=self.dec(z)
        return out


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))* 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()
