#!/usr/bin/env python
import argparse
import pickle

from torchvision import transforms

import modules.training_methods as run_train
import modules.training_methods_safn as run_train_safn
import modules.transforms as RGBDtransforms
from modules.datasets import SynROD_ROD
from modules.datasets import TransformedDataset
from modules.net import Net, AFNNet


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="")
    parser.add_argument("--ram", dest='ram', action='store_false')
    parser.add_argument("--no-ram", dest='ram', action='store_false')
    parser.set_defaults(ram=True)
    parser.add_argument("--ckpt_dir", default="none",
                        help='select --ckpt_dir=none if not desired')
    parser.add_argument("--result_dir", default="output/")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=0.0003)
    parser.add_argument("--class_num", default=47, type=int)
    parser.add_argument("--dropout_p", default=0.5)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--weight_decay", default=0.05)
    parser.add_argument("--step_size", default=10)
    parser.add_argument("--gamma", default=1)
    parser.add_argument("--lambda", dest='lamda', default=1.0, help='weight for pretext loss')
    parser.add_argument("--entropy_weight", default=1, help='weight for entropy loss')
    parser.add_argument("--dr", default=1.0, help='step size for SAFN')
    parser.add_argument("--radius", default=25, help='shared fixed R value for HAFN')
    parser.add_argument("--weight_L2norm", default=0.05, help='weight of AFN loss')
    parser.add_argument("--experiment", default='rr', choices=['rr', 'safn', 'safn-rr'],
                        help='select the experiment to run:'
                             '`rr`: domain adaptation with relative rotation,'
                             '`safn`: stepwise afn,'
                             '`safn-rr`: stepwise afn and relative rotation DA')

    args = parser.parse_args()
    return args


def load_datasets():
    global source_train_dataset_main
    global source_train_dataset_pretext

    global source_test_dataset_main
    global source_test_dataset_pretext

    global target_test_dataset_main
    global target_dataset_pretext
    global target_train_dataset_main

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

    source_train_dataset = SynROD_ROD(args.data_root, category="synROD", RAM=args.ram, split="train")
    source_test_dataset = SynROD_ROD(args.data_root, category="synROD", RAM=args.ram, split="test")
    target_dataset = SynROD_ROD(args.data_root, category="ROD", RAM=args.ram)

    source_train_dataset_main = TransformedDataset(source_train_dataset, train_transform)
    source_train_dataset_pretext = TransformedDataset(source_train_dataset, train_transform_rotation)

    source_test_dataset_main = TransformedDataset(source_test_dataset, val_transform)
    source_test_dataset_pretext = TransformedDataset(source_test_dataset, val_transform_rotation)

    # Data loader for ROD train and test - PRETEXT at train, MAIN at test (check validity of drop last when testing)
    target_test_dataset_main = TransformedDataset(target_dataset, val_transform)
    target_dataset_pretext = TransformedDataset(target_dataset, val_transform_rotation)
    target_train_dataset_main = TransformedDataset(target_dataset, train_transform)


def train():
    checkpoint_dir = None if str.lower(args.ckpt_dir) == 'none' else args.ckpt_dir

    if args.experiment == 'rr':
        net = Net(args.class_num)
        results = run_train.train_RGBD_DA(net,
                                          source_train_dataset_main,
                                          source_train_dataset_pretext,
                                          target_test_dataset_main,
                                          target_dataset_pretext,
                                          source_test_dataset_main,
                                          source_test_dataset_pretext,
                                          args.batch_size, args.epochs, args.lr, args.momentum,
                                          args.step_size, args.gamma, args.entropy_weight, args.lamda,
                                          checkpoint_dir, args.weight_decay,
                                          target_train_dataset_main)
    elif args.experiment == 'safn':
        net = AFNNet(args.class_num)
        results = run_train_safn.RGBD_e2e_SAFN(net,
                                               source_train_dataset_main,
                                               target_train_dataset_main,
                                               source_test_dataset_main,
                                               target_test_dataset_main,
                                               args.batch_size, args.epochs, args.lr, args.momentum,
                                               args.step_size, args.gamma, checkpoint_dir,
                                               args.weight_decay, args.dr,
                                               args.weight_L2norm, True, args.entropy_weight)
    elif args.experiment == 'safn-rr':
        net = AFNNet(args.class_num)
        results = run_train_safn.train_RGBD_DA_SAFN(net, source_train_dataset_main,
                                                    source_train_dataset_pretext,
                                                    target_test_dataset_main,
                                                    target_dataset_pretext,
                                                    target_train_dataset_main,
                                                    source_test_dataset_main, args.batch_size, args.epochs, args.lr,
                                                    args.momentum, args.step_size, args.gamma, args.entropy_weight,
                                                    args.lamda, args.ckpt_dir, args.weight_decay, args.dr,
                                                    args.weight_L2norm)
    else:
        print(f'ERROR: Experiment {args.experiment} not available')
        raise ValueError

    res_file = open(args.result_dir, 'wb')
    pickle.dump(results, res_file)


if __name__ == '__main__':
    args = parse_arg()

    print('Loading datasets...')
    load_datasets()

    print(f'Training network with method {args.experiment}')
    train()
