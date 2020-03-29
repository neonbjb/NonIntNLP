import os
import torchvision
import torch
import torch.nn as nn
import torch.utils.data

# Root directory for dataset
dataroot = "C:\\Users\\jbetk\\Documents\\data\\ml\\celeba"
# Directory to save checkpoints
output_dir = "C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer"
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 20
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Whether or not to log to w&b
do_wandb = True

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
for g in range(0, 30000, 2000):
    netG = torch.load(os.path.join(output_dir, "generator_%i.pt" % g))
    with torch.no_grad():
        imglist = []
        num_fakes = 50
        noise = torch.randn(num_fakes, nz, 1, 1, device=device)
        fakes = netG(noise).detach().cpu()
        fakes_dir = os.path.join(output_dir, "fakes_%i" % g)
        if not os.path.exists(fakes_dir):
            os.makedirs(fakes_dir)
        for i in range(num_fakes):
            torchvision.utils.save_image(fakes[i], os.path.join(fakes_dir, "%i.png" % i))