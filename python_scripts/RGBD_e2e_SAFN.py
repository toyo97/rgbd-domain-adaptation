#!/usr/bin/env python
import time

import torch
from torchvision import transforms


def main():
    since = time.time()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 47
    WEIGHT_DECAY = 0.05

    NUM_EPOCHS = 40

    LR = 0.0003
    MOMENTUM = 0.9
    STEP_SIZE = 10
    GAMMA = 1
    ENTROPY_WEIGHT = 0.1
    BATCH_SIZE = 64

    DELTAR = 1.0
    WEIGHT_L2NORM = 0.05
    ENTROPY = True

    DATA_DIR = 'repo/rgbd-domain-adaptation.git/trunk'  # 'rgbd'

    from modules.datasets import TransformedDataset
    from modules.net import AFNNet
    import modules.transforms as RGBDtransforms
    import modules.training_methods_safn as run_train
    from modules.datasets import SynROD_ROD

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

    source_test_dataset_main = TransformedDataset(source_test_dataset, val_transform)

    # Data loader for ROD train and test - PRETEXT at train, MAIN at test (check validity of drop last when testing)
    target_train_dataset_main = TransformedDataset(target_dataset, train_transform)
    target_test_dataset_main = TransformedDataset(target_dataset, val_transform)

    net = AFNNet(NUM_CLASSES, dropout_p=0.5)
    time_elapsed = time.time() - since
    print('Time to create dataset: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    train_losses, val_losses, train_accs, val_accs = run_train.RGBD_e2e_SAFN(net,
                                                                             source_train_dataset_main,
                                                                             target_train_dataset_main,
                                                                             source_test_dataset_main,
                                                                             target_test_dataset_main,
                                                                             BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM,
                                                                             STEP_SIZE, GAMMA, 'checkpoints/hafn/e2e',
                                                                             WEIGHT_DECAY,
                                                                             dr=DELTAR, weight_L2norm=WEIGHT_L2NORM,
                                                                             entropy=ENTROPY, entropy_weight=ENTROPY_WEIGHT)


if __name__ == "__main__":
    main()
