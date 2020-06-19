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
    from modules.net import Net, AFNNet
    import modules.transforms as RGBDtransforms
    import modules.training_methods as run_train
    import modules.training_methods_safn as run_train_safn
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
    DR = 1
    LR = 0.0003
    MOMENTUM = 0.9
    STEP_SIZE = 10
    GAMMA = 1
    LAMBDA = 1
    ENTROPY_WEIGHT = 0.1

    WEIGHT_L2NORM = 0.05
    BATCH_SIZE = 32

    param_grid = ParameterGrid([
            {"weight_decay": [0.05, 0.005, 0.0005],
             'lr': np.logspace(-2, -5, 50),
             'lamda': [1, 0.8, 0.5],
             'weight_L2norm': [0.1, 0.05, 0.01],
             'entropy_weight': [0.1, 0.2, 0.05]
            #,'step_size': np.arange(2, 8),
            #'gamma': [0.3, 0.1, 0.05, 0.02]
            }
        ])

    params_list = random.sample(list(param_grid), 10)

    for i, params in enumerate(params_list):
        net = AFNNet(NUM_CLASSES)
        state_dict = {'params': params}
        # results = train_losses, val_losses, train_accs, val_accs
        state_dict['results'] = run_train_safn.train_RGBD_DA_SAFN(net, source_train_dataset_main, source_train_dataset_pretext, target_dataset_main,
                                                                  target_dataset_pretext, target_dataset_main_entropy_loss, source_test_dataset_main, BATCH_SIZE,
                                                                  NUM_EPOCHS, params["lr"], MOMENTUM, STEP_SIZE, GAMMA, params["entropy_weight"], params["lamda"], None,
                                                                  params["weight_decay"], DR, params["weight_L2norm"])
        res_file = open(f'tuning/SAFN_RR/res_{i}.obj', 'wb')
        pickle.dump(state_dict, res_file)
        # LOAD
        # file_pi2 = open('filename_pi.obj', 'r')
        # object_pi2 = pickle.load(file_pi2)


if __name__ == '__main__':
    tuning()
