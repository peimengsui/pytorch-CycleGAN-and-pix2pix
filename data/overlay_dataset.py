import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random

from DataUtils.add_noise_to_image import alpha_blend
from natsort import natsorted
import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte


class OverlayDataset(BaseDataset):
  def initialize(self, opt):
    self.opt = opt
    self.root = opt.dataroot
    self.dir_orig = os.path.join(opt.dataroot, opt.phase, 'original')
    self.dir_noise = os.path.join(opt.dataroot, opt.phase, 'texture')

    self.orig_paths = make_dataset(self.dir_orig)
    self.noise_paths = make_dataset(self.dir_noise)

    self.orig_paths = natsorted(self.orig_paths)
    self.noise_paths = natsorted(self.noise_paths)
    self.orig_permute_paths = np.arange(len(self.orig_paths))
    if opt.phase == 'train':
      self.orig_permute_paths = np.random.permutation(len(self.orig_paths))
    self.orig_size = len(self.orig_paths)
    self.transform = get_transform(opt)
    self.alpha = opt.alpha
    self.imagemode = opt.imagemode

  def __getitem__(self, index):
    orig_path = self.orig_paths[index % self.orig_size]
    index_orig = index % self.orig_size
    orig_pair_path = self.orig_paths[self.orig_permute_paths[index_orig]]
    noise_path = self.noise_paths[self.orig_permute_paths[index_orig]]
    #print('(A, B) = (%d, %d)' % (index_A, index_B))
    if self.imagemode == 'RGB':
      orig_img = Image.open(orig_path).convert('RGB')
      noise_img = Image.open(noise_path).convert('RGB')
      orig_pair_img = Image.open(orig_pair_path).convert('RGB')
    elif self.imagemode == 'L':
      orig_img = Image.open(orig_path).convert('L')
      noise_img = Image.open(noise_path).convert('L')
      orig_pair_img = Image.open(orig_pair_path).convert('L')
    A_img = self.transform(orig_img)
    blended_img = alpha_blend(np.array(noise_img), np.array(orig_pair_img), self.alpha)
    blended_img = img_as_ubyte(blended_img)
    B_img = self.transform(Image.fromarray(blended_img))

    return {'A': A_img, 'B': B_img,
            'A_paths': orig_path, 'B_paths': orig_pair_path}

  def __len__(self):
    return self.orig_size

  def name(self):
    return 'OverlayDataset'
