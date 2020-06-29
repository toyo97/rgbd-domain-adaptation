#!/usr/bin/env python
import random
from torchvision import transforms
import time
from sklearn.model_selection import ParameterGrid
import numpy as np
import pickle


def tuning():
    since = time.time()

    DATA_DIR = 'repo/rgbd-domain-adaptation.git/trunk'  # 'rgbd'

    from modules.datasets import TransformedDataset
    from modules.net import Net, AFNNet, ablationAFNNet
    import modules.transforms as RGBDtransforms
    import modules.training_methods as run_train
    import modules.training_methods_safn as run_train_safn
    import modules.training_methods_hafn as run_train_hafn
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
    RAMFlag = True
    source_train_dataset = SynROD_ROD(DATA_DIR, category="synROD", RAM=RAMFlag, split="train")
    source_test_dataset = SynROD_ROD(DATA_DIR, category="synROD", RAM=RAMFlag, split="test")
    target_dataset = SynROD_ROD(DATA_DIR, category="ROD", RAM=RAMFlag)

    source_train_dataset_main = TransformedDataset(source_train_dataset, train_transform)
    source_train_dataset_pretext = TransformedDataset(source_train_dataset, train_transform_rotation)

    source_test_dataset_main = TransformedDataset(source_test_dataset, val_transform)
    source_test_dataset_pretext = TransformedDataset(source_test_dataset, val_transform_rotation)

    # Data loader for ROD train and test - PRETEXT at train, MAIN at test (check validity of drop last when testing)
    target_dataset_main = TransformedDataset(target_dataset, val_transform)
    target_dataset_pretext = TransformedDataset(target_dataset, val_transform_rotation)
    target_dataset_main_entropy_loss = TransformedDataset(target_dataset, train_transform)

    time_elapsed = time.time() - since
    print('Time to create dataset: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    NUM_CLASSES = 47
    WEIGHT_DECAY = 0.05

    NUM_EPOCHS = 10

    LR = 0.0003
    MOMENTUM = 0.9
    STEP_SIZE = 10
    GAMMA = 1

    BATCH_SIZE = 64

    LAMBDA = 1
    ENTROPY_WEIGHT = 0.1

    WEIGHT_L2NORM = 0.05
    DR = 1
    RADIUS = 25

    # HAFN e2e 5 final run

    params = {'gamma': 0.05, 'lr': 0.0005179474679231213, 'step_size': 2}

    BATCH_SIZE =  32
    for run in range(5):
        net = AFNNet(NUM_CLASSES)
        state_dict = {'params': params}
        state_dict['results'] = run_train_hafn.RGBD_e2e_HAFN(net,
                                                             source_train_dataset_main,
                                                             target_dataset_main_entropy_loss,
                                                             source_test_dataset_main,
                                                             target_dataset_main,
                                                             BATCH_SIZE, NUM_EPOCHS, params["lr"], MOMENTUM,
                                                             params["step_size"], params["gamma"], None, WEIGHT_DECAY,
                                                             RADIUS, WEIGHT_L2NORM, 0.5)

        res_file = open(f'final_results/HAFNe2e5runs/res_{i}.obj', 'wb')
        pickle.dump(state_dict, res_file)

    # SAFN RGB/ depth / e2e without rescaling factor after dropout

    BATCH_SIZE = 32

    net = AFNNet(NUM_CLASSES, single_mod="RGB", rescale_dropout= False)
    params = {'used': 'default params'}
    state_dict = {'params': params}
    state_dict['results'] = run_train_safn.train_sourceonly_singlemod_SAFN(net, "RGB",
                                    source_train_dataset_main,
                                    target_dataset_main_entropy_loss,
                                    source_test_dataset_main,
                                    target_dataset_main,
                                    BATCH_SIZE, LR, MOMENTUM, STEP_SIZE, GAMMA, NUM_EPOCHS, None,
                                    WEIGHT_DECAY,
                                    DR, WEIGHT_L2NORM, True, ENTROPY_WEIGHT)

    res_file = open(f'final_results/normal_dropout_SAFN/RGB/res_{i}.obj', 'wb')
    pickle.dump(state_dict, res_file)

    params = {'gamma': 0.1, 'lr': 0.0037, 'step_size': 7}
    BATCH_SIZE = 32

    net = AFNNet(NUM_CLASSES, single_mod="depth", rescale_dropout=False)

    state_dict = {'params': params}
    state_dict['results'] = run_train_safn.train_sourceonly_singlemod_SAFN(net, "depth",
                                                                           source_train_dataset_main,
                                                                           target_dataset_main_entropy_loss,
                                                                           source_test_dataset_main,
                                                                           target_dataset_main,
                                                                           BATCH_SIZE, params["lr"], MOMENTUM, params["step_size"], params["gamma"],
                                                                           NUM_EPOCHS, None,
                                                                           WEIGHT_DECAY,
                                                                           DR, WEIGHT_L2NORM, True, ENTROPY_WEIGHT)

    res_file = open(f'final_results/normal_dropout_SAFN/RGB/res_{i}.obj', 'wb')
    pickle.dump(state_dict, res_file)

    params = {'gamma': 0.3, 'lr': 0.0024, 'step_size': 7}
    BATCH_SIZE = 32

    net = AFNNet(NUM_CLASSES, rescale_dropout=False)

    state_dict = {'params': params}
    state_dict['results'] = run_train_safn.RGBD_e2e_SAFN(net,
                  source_train_dataset_main,
                  target_dataset_main_entropy_loss,
                  source_test_dataset_main,
                  target_dataset_main,
                  BATCH_SIZE, NUM_EPOCHS, params["lr"],  MOMENTUM, params["step_size"], params["gamma"], None, WEIGHT_DECAY,
                  DR, WEIGHT_L2NORM, True, ENTROPY_WEIGHT)

    res_file = open(f'final_results/normal_dropout_SAFN/e2e/res_{i}.obj', 'wb')
    pickle.dump(state_dict, res_file)


if __name__ == '__main__':
    tuning()
