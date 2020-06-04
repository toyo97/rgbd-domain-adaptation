import torchvision.transforms.functional as TF
import random


def coupled_rotation(rgb, depth):

  j = random.randint(0, 3)
  k = random.randint(0, 3)

  z = (k-j) % 4

  # Note: TF.rotate is counter-clockwise
  rgb_new = TF.rotate(rgb, 270*j)
  depth_new = TF.rotate(depth, 270*k)

  return rgb_new, depth_new, z

def coupled_crop(rgb, depth):
  # TODO implement
  pass

def coupled_hflip(rgb, depth):
  # TODO implement
  pass

# Here the two transform functions to be passed in the dataset parameter
def pretext_transform(rgb, depth):
  # TODO combine the transformations for the pretext-task datasets
  pass

def main_transform(rgb, depth):
  # TODO combine transformations for the main-task datasets
  pass