import argparse
import os
import pickle
import sys
import time

import torch
from torch import optim, nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import modules.transforms as RGBDtransforms
from modules.datasets import SynROD_ROD
from modules.datasets import TransformedDataset
from modules.net import Net
from modules.training_methods import load_checkpoint


def validate(net, loader):
    tot_loss = 0
    running_corrects = 0
    for images_rgb, images_d, labels in loader:
        images_rgb = images_rgb.to(device)
        images_d = images_d.to(device)
        labels = labels.to(device)

        outputs = net(images_rgb, images_d)
        loss = criterionFinalLoss(outputs, labels)
        tot_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data).data.item()
    tot_loss = tot_loss / float(len(src_val_m_ds))
    tot_acc = running_corrects / float(len(src_val_m_ds))
    return tot_loss, tot_acc


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="")
parser.add_argument("--ram", default="y")
parser.add_argument("--ckpt_dir", default="checkpoints/source_only/e2e",
                    help='select --ckpt_dir=none if not desired')
parser.add_argument("--result_dir", default="result/source_only/e2e")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--lr", default=0.0003)
parser.add_argument("--class_num", default=47, type=int)
parser.add_argument("--dropout_p", default=0.5)
parser.add_argument("--momentum", default=0.9)
parser.add_argument("--weight_decay", default=0.05)
parser.add_argument("--step_size", default=10)
parser.add_argument("--gamma", default=1)
args = parser.parse_args()
ram = True if args.ram == "y" else False

since = time.time()

train_transform = RGBDtransforms.RGBDCompose([transforms.Resize((256, 256)),
                                              RGBDtransforms.CoupledRandomCrop(224),  # random crop for training
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                              ])

val_transform = RGBDtransforms.RGBDCompose([transforms.Resize((256, 256)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])
                                            ])
# source/target train/validation datasets
src_tr_ds = SynROD_ROD(args.data_root, category="synROD", RAM=ram, split="train")
src_val_ds = SynROD_ROD(args.data_root, category="synROD", RAM=ram, split="test")
tgt_ds = SynROD_ROD(args.data_root, category="ROD", RAM=ram)

# dataset with main-head transformation
src_tr_m_ds = TransformedDataset(src_tr_ds, train_transform)
src_val_m_ds = TransformedDataset(src_val_ds, val_transform)
tgt_m_ds = TransformedDataset(tgt_ds, val_transform)

time_elapsed = time.time() - since
print('Time to create dataset: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# dataloaders for source/target train/validation sets
src_tr_m_dl = DataLoader(src_tr_m_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, drop_last=True)
tgt_val_dl = DataLoader(tgt_m_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        drop_last=False)
src_val_m_dl = DataLoader(src_val_m_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, drop_last=False)

# instantiate model
net = Net(args.class_num)
net = net.to(device)

# load checkpoint if available
if args.ckpt_dir != "none":
    checkpoint = load_checkpoint(args.ckpt_dir)
else:
    checkpoint = None

# initialize training objects
if checkpoint is not None:
    net.load_state_dict(checkpoint['net'])
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch0 = checkpoint['epoch'] + 1  # starting from the next epoch
    source_losses = checkpoint['source_losses']
    source_accs = checkpoint['source_accs']
    target_losses = checkpoint['target_losses']
    target_accs = checkpoint['target_accs']
    print(f'Checkpoint found! Starting from epoch {epoch0}')
else:
    epoch0 = 0
    source_losses = []
    source_accs = []
    target_losses = []
    target_accs = []
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(f'No checkpoint found, starting from epoch 1')

criterion = nn.CrossEntropyLoss()
criterionFinalLoss = nn.CrossEntropyLoss(reduction='sum')

cudnn.benchmark = True

for epoch in range(epoch0, args.epochs):
    print(f'Epoch {epoch + 1}/{args.epochs}')
    since1 = time.time()
    running_loss_m = 0.0

    net.train()
    it = 0
    for rimgs, dimgs, labels in src_tr_m_dl:

        optimizer.zero_grad()

        # Bring data over the device of choice
        rimgs = rimgs.to(device)
        dimgs = dimgs.to(device)
        labels = labels.to(device)

        # forward
        outputs = net(rimgs, dimgs)
        # compute main loss
        loss_m = criterion(outputs, labels)
        loss_m.backward()

        # update weights
        optimizer.step()

        # print statistics
        running_loss_m += loss_m.item()
        if it % 100 == 99:  # print every 100 mini-batches
            print(f'[{epoch + 1}, {it + 1}] Lm {running_loss_m / 100}')
            running_loss_m = 0.
        it += 1

    net.train(False)
    # validate source
    src_loss, src_acc = validate(net, src_val_m_dl)
    source_losses.append(src_loss)
    source_accs.append(src_acc)

    # validate target
    tgt_loss, tgt_acc = validate(net, tgt_val_dl)
    target_losses.append(tgt_loss)
    target_accs.append(tgt_acc)

    scheduler.step()

    # CHECKPOINT
    if args.ckpt_dir != "none":
        filename = str(epoch + 1) + '.ckpt'
        path = os.path.join(args.ckpt_dir, filename)
        torch.save({
            'epoch': epoch,
            'source_losses': source_losses,
            'source_accs': source_accs,
            'target_losses': target_losses,
            'target_accs': target_accs,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, path)
        print(f'Checkpoint {epoch + 1} saved succesfully')

    elapsed = time.time() - since1
    print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))

result = source_losses, target_losses, source_accs, target_accs
# save the results
pickle.dump(result, open(args.result_dir, 'wb'))
