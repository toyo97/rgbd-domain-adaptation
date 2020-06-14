#!/usr/bin/env python
import time

from torchvision import transforms

import modules.training_methods_safn as run_train
import modules.transforms as RGBDtransforms
from modules.datasets import SynROD_ROD
from modules.datasets import TransformedDataset
from modules.net import AFNNet


def main():
    since = time.time()

    NUM_CLASSES = 47
    WEIGHT_DECAY = 0.05

    NUM_EPOCHS = 40

    LR = 0.0003
    MOMENTUM = 0.9
    STEP_SIZE = 10
    GAMMA = 1

    BATCH_SIZE = 64

    LAMBDA = 1  # weights contribution of the pretext loss to the total objective
    ENTROPY_WEIGHT = 0.1

    DELTAR = 1.0
    WEIGHT_L2NORM = 0.05

    DATA_DIR = 'repo/rgbd-domain-adaptation.git/trunk'  # 'rgbd'

    imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = RGBDtransforms.RGBDCompose([transforms.Resize((256, 256)),
                                                  RGBDtransforms.CoupledRandomCrop(224),  # random crop for training
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=imgnet_mean,  # ImageNet mean and std
                                                                       std=imgnet_std)]
                                                 )

    train_transform_rotation = RGBDtransforms.RGBDCompose([transforms.Resize((256, 256)),
                                                           RGBDtransforms.CoupledRandomCrop(224),
                                                           # random crop for training
                                                           RGBDtransforms.CoupledRotation(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=imgnet_mean,
                                                                                # ImageNet mean and std
                                                                                std=imgnet_std)]
                                                          )

    val_transform = RGBDtransforms.RGBDCompose([transforms.Resize((256, 256)),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=imgnet_mean,
                                                                     std=imgnet_std)]
                                               )

    val_transform_rotation = RGBDtransforms.RGBDCompose([transforms.Resize((256, 256)),
                                                         transforms.CenterCrop(224),
                                                         RGBDtransforms.CoupledRotation(),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=imgnet_mean,
                                                                              std=imgnet_std)]
                                                        )
    source_train_dataset = SynROD_ROD(DATA_DIR, category="synROD", RAM=True, split="train")
    source_test_dataset = SynROD_ROD(DATA_DIR, category="synROD", RAM=True, split="test")
    target_dataset = SynROD_ROD(DATA_DIR, category="ROD", RAM=True)

    source_train_dataset_main = TransformedDataset(source_train_dataset, train_transform)
    source_train_dataset_pretext = TransformedDataset(source_train_dataset, train_transform_rotation)

    source_test_dataset_main = TransformedDataset(source_test_dataset, val_transform)

    target_dataset_main = TransformedDataset(target_dataset, val_transform)
    target_dataset_main_ent = TransformedDataset(target_dataset, train_transform)
    target_dataset_pretext = TransformedDataset(target_dataset, train_transform_rotation)

    net = AFNNet(NUM_CLASSES)
    time_elapsed = time.time() - since
    print('Time to create dataset: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    train_losses, val_losses, train_accs, val_accs = run_train.train_RGBD_DA_SAFN(net, source_train_dataset_main,
                                                                                  source_train_dataset_pretext,
                                                                                  target_dataset_main,
                                                                                  target_dataset_pretext,
                                                                                  target_dataset_main_ent,
                                                                                  source_test_dataset_main, BATCH_SIZE,
                                                                                  NUM_EPOCHS, LR, MOMENTUM, STEP_SIZE,
                                                                                  GAMMA, ENTROPY_WEIGHT, LAMBDA,
                                                                                  'checkpoints/RGB_DA_SAFN',
                                                                                  WEIGHT_DECAY, DELTAR, WEIGHT_L2NORM)


if __name__ == "__main__":
    main()
