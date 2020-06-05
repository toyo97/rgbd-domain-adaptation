import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from torch.utils.data.dataset import random_split
from torch.backends import cudnn
import time
import os
from getpass import getpass
import urllib
from torch.utils.data import DataLoader

# forse non servono questi import
import tqdm
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
from PIL import Image
import os
import os.path

# forse non servono questi import
from torchvision import models
from torch.autograd import Function
import copy


if __name__=='__main__':
  since = time.time()
  
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  NUM_CLASSES = 47
  WEIGHT_DECAY = 0.05

  NUM_EPOCHS = 40

  LR = 0.0003
  MOMENTUM = 0.9
  STEP_SIZE = 10
  GAMMA = 0.1

  BATCH_SIZE = 64
  MODALITY = "RGB"

  LAMBDA = 1 # weights contribution of the pretext loss to the total objective
  ENTROPY_WEIGHT = 0.1
  
  
  if not os.path.isdir('./rgbd'):
    user = #scrivi username di Github
    password = #scrivi password di Github
    password = urllib.parse.quote(password)

    cmd_string = 'git clone https://{0}:{1}@github.com/toyo97/rgbd-domain-adaptation.git'.format(user, password)

    os.system(cmd_string)
    cmd_string, password = "", "" # removing the password from the variable
    os.system("mv rgbd-domain-adaptation rgbd")
    os.systen("mkdir modules")
    os.system("cp -r rgbd/modules/ modules/")
  else:
    # update code changes
    os.system("git -C rgbd/ pull")
    os.system("cp -ur rgbd/modules/ modules/")

  time_elapsed = time.time() - since
  print('Time to create dataset: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  DATA_DIR = 'rgbd'
  
  from modules.modules.datasets import TransformedDataset
  from modules.modules.net import Net
  import modules.modules.transforms as RGBDtransforms
  import modules.modules.training_methods as run_train
  from modules.modules.datasets import SynROD_ROD
  
  
  imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
  train_transform = RGBDtransforms.RGBDCompose([transforms.Resize((256,256)),
                                              RGBDtransforms.CoupledRandomCrop(224), # random crop for training
                                              transforms.ToTensor(),                                     
                                              transforms.Normalize( mean=imgnet_mean, # ImageNet mean and std
                                                                    std=imgnet_std)]
  )

  train_transform_rotation = RGBDtransforms.RGBDCompose([transforms.Resize((256,256)),
                                                RGBDtransforms.CoupledRandomCrop(224), # random crop for training
                                                RGBDtransforms.CoupledRotation(),
                                                transforms.ToTensor(),                                     
                                                transforms.Normalize( mean=imgnet_mean, # ImageNet mean and std
                                                                      std=imgnet_std)]
  )

  val_transform = RGBDtransforms.RGBDCompose([transforms.Resize((256,256)),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize( mean=imgnet_mean,
                                                                    std=imgnet_std)]
  )

  val_transform_rotation = RGBDtransforms.RGBDCompose([transforms.Resize((256,256)),
                                              transforms.CenterCrop(224),
                                              RGBDtransforms.CoupledRotation(),
                                              transforms.ToTensor(),
                                              transforms.Normalize( mean=imgnet_mean,
                                                                    std=imgnet_std)]
  )
  source_train_dataset = SynROD_ROD(DATA_DIR, category="synROD", RAM=False, split="train")
  source_test_dataset = SynROD_ROD(DATA_DIR, category="synROD", RAM=False, split="test")
  target_dataset = SynROD_ROD(DATA_DIR, category="ROD", RAM = False)
  
  source_train_dataset_main = TransformedDataset(source_train_dataset, train_transform)
  source_train_dataset_pretext = TransformedDataset(source_train_dataset, train_transform_rotation)

  source_test_dataset_main = TransformedDataset(source_test_dataset, val_transform)
  source_test_dataset_pretext = TransformedDataset(source_test_dataset, val_transform_rotation)

  # Data loader for ROD train and test - PRETEXT at train, MAIN at test (check validity of drop last when testing)
  target_dataset_main = TransformedDataset(target_dataset, val_transform)
  target_dataset_pretext = TransformedDataset(target_dataset, val_transform_rotation)
  
  net = Net(NUM_CLASSES)
  
  train_losses, val_losses, train_accs, val_accs = run_train.train_sourceonly_singlemod(net, MODALITY,
                               source_train_dataset_main, source_test_dataset_main,
                               target_dataset_main, BATCH_SIZE, LR, MOMENTUM, STEP_SIZE, GAMMA, NUM_EPOCHS, checkpoint='local')
  
