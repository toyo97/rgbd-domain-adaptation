#!/usr/bin/env python
import random
from torchvision import transforms
import time
from sklearn.model_selection import ParameterGrid
import numpy as np
import pickle


def tuning():
    since = time.time()

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

    params = {"dr":1, "weight_decay": WEIGHT_DECAY, "lr": LR, "entropy_weight": ENTROPY_WEIGHT, "weight_l2norm": WEIGHT_L2NORM, "batch_size": BATCH_SIZE}
    MODALITY = "RGB"

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

    for modality in ['RGB', 'depth']:
        net = AFNNet(NUM_CLASSES, modality)
        state_dict = {'params': params}
        # results = train_losses, val_losses, train_accs, val_accs
        state_dict['results'] = run_train_safn.train_sourceonly_singlemod(net, modality,
                                    source_train_dataset_main,
                                    target_dataset_main_entropy_loss,
                                    source_test_dataset_main,
                                    target_dataset_main,
                                    BATCH_SIZE, LR, MOMENTUM, STEP_SIZE, GAMMA, 20, None,
                                    WEIGHT_DECAY,
                                    DR, WEIGHT_L2NORM, True, ENTROPY_WEIGHT)

        res_file = open(f'tuning/only_SAFN/{modality}/DEFAULT/default_params.obj', 'wb')
        pickle.dump(state_dict, res_file)
        # LOAD
        # file_pi2 = open('filename_pi.obj', 'r')
        # object_pi2 = pickle.load(file_pi2)


    net = AFNNet(NUM_CLASSES)
    state_dict = {'params': params}
    state_dict['results'] = run_train_safn.RGBD_e2e_SAFN(net,
                                                        source_train_dataset_main,
                                                        target_dataset_main_entropy_loss,
                                                        source_test_dataset_main,
                                                        target_dataset_main,
                                                        BATCH_SIZE, 20, LR, MOMENTUM, STEP_SIZE, GAMMA, None, WEIGHT_DECAY,
                                                        DR, WEIGHT_L2NORM, True, ENTROPY_WEIGHT)

    res_file = open(f'tuning/only_SAFN/e2e/DEFAULT/default_params.obj', 'wb')
    pickle.dump(state_dict, res_file)



    """param_grid = ParameterGrid([
        {"weight_decay": [0.05, 0.005, 0.0005]}
    ])

    params_list = random.sample(list(param_grid), 10)
    for modality in ['RGB', 'depth']:
        for i, params in enumerate(params_list):
            net = Net(NUM_CLASSES, modality)
            state_dict = {'params': params}
            # results = train_losses, val_losses, train_accs, val_accs
            state_dict['results'] = run_train.train_sourceonly_singlemod(net, modality, source_train_dataset_main,
                                                                         source_test_dataset_main,
                                                                         target_dataset_main,
                                                                         BATCH_SIZE, params['lr'], MOMENTUM,
                                                                         params['step_size'], params['gamma'],
                                                                         NUM_EPOCHS,
                                                                         None,
                                                                         WEIGHT_DECAY)

            res_file = open(f'tuning/source_only_{modality}/res_{i}.obj', 'wb')
            pickle.dump(state_dict, res_file)
            # LOAD
            # file_pi2 = open('filename_pi.obj', 'r')
            # object_pi2 = pickle.load(file_pi2)

    # E2E
    for i, params in enumerate(params_list):
        net = Net(NUM_CLASSES)
        state_dict = {'params': params}
        state_dict['results'] = run_train.RGBD_e2e(net,
                                                                     source_train_dataset_main,
                                                                     target_dataset_main,
                                                                     source_test_dataset_main,
                                                                     BATCH_SIZE, NUM_EPOCHS, params['lr'], MOMENTUM, params['step_size'],
                                                                     params['gamma'], None, WEIGHT_DECAY)

        res_file = open(f'tuning/source_only_e2e/res_{i}.obj', 'wb')
        pickle.dump(state_dict, res_file)"""

    NUM_CLASSES = 47
    WEIGHT_DECAY = 0.05

    NUM_EPOCHS = 10

    LR = 0.0003
    MOMENTUM = 0.9
    STEP_SIZE = 10
    GAMMA = 1
    LAMBDA = 1
    ENTROPY_WEIGHT = 0.1

    BATCH_SIZE = 64


    for i, wd in enumerate([0.05, 0.005, 0.0005]):
        net = Net(NUM_CLASSES)
        state_dict = {'wd': wd}
        # results = train_losses, val_losses, train_accs, val_accs
        state_dict['results'] = run_train.train_RGBD_DA(net,
                                                        source_train_dataset_main, source_train_dataset_pretext,
                                                        target_dataset_main, target_dataset_pretext,
                                                        source_test_dataset_main, source_test_dataset_pretext,
                                                        BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, STEP_SIZE, GAMMA,
                                                        ENTROPY_WEIGHT, LAMBDA, None,
                                                        wd, target_dataset_main_entropy_loss)

        res_file = open(f'tuning/RGBD_DA_RR/weight_decay/res{i}.obj', 'wb')
        pickle.dump(state_dict, res_file)
        # LOAD
        # file_pi2 = open('filename_pi.obj', 'r')
        # object_pi2 = pickle.load(file_pi2)



if __name__ == '__main__':
    tuning()
