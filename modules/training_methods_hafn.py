import glob
import os
import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader


def entropy_loss(logits):
    p_softmax = F.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to, used for numerical stability
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))


def loopy(dl):
    """
  Allow iterating over a dataset more than once
  to deal with different number of samples between datasets
  during training and batch sampling
  :param dl: dataloader
  """
    while True:
        for x in dl: yield x


def load_checkpoint(checkpoint_dir):
    """
  Return last state dictionary if available, i.e. if training ended prematurely
  :param checkpoint_dir:
  """
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        return torch.load(latest_file)
    else:
        # no checkpoint available
        return None


def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss


def get_L2norm_loss_self_driven(x, radius):
    l = (x.norm(p=2, dim=1).mean() - radius) ** 2
    return l


def RGBD_e2e_HAFN(net,
                  source_train_dataset_main,
                  target_train_dataset_main,
                  source_test_dataset_main,
                  target_test_dataset_main,
                  batch_size, num_epochs, lr, momentum, step_size, gamma, checkpoint_dir, weight_decay,
                  radius, weight_L2norm, dropout_p):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    # Load checkpoint if available
    checkpoint = None
    if checkpoint_dir is not None:
        checkpoint = load_checkpoint(checkpoint_dir)
    if checkpoint is not None:
        net.load_state_dict(checkpoint['net'])

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if checkpoint is not None:
        epoch0 = checkpoint['epoch'] + 1  # starting from the next epoch
        source_losses = checkpoint['source_losses']
        source_accs = checkpoint['source_accs']
        target_losses = checkpoint['target_losses']
        target_accs = checkpoint['target_accs']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        print(f'Checkpoint found! Starting from epoch {epoch0}')

    else:
        epoch0 = 0
        source_losses = []
        source_accs = []
        target_losses = []
        target_accs = []

        print(f'No checkpoint found, starting from epoch 1')

    # Data loaders for training phase
    # SOURCE
    source_train_main_dataloader = DataLoader(source_train_dataset_main, batch_size=batch_size, shuffle=True,
                                              num_workers=1, drop_last=True)

    # TARGET TRAIN
    target_train_main_dataloader = DataLoader(target_train_dataset_main, batch_size=batch_size, shuffle=True,
                                              num_workers=1, drop_last=True)

    # TARGET
    target_validation_dataloader = DataLoader(target_test_dataset_main, batch_size=batch_size, shuffle=True,
                                              num_workers=1,
                                              drop_last=False)

    # used in validation (drop_last = False)
    source_test_main_dataloader = DataLoader(source_test_dataset_main, batch_size=batch_size, shuffle=True,
                                             num_workers=1,
                                             drop_last=False)

    criterion = nn.CrossEntropyLoss()
    criterionFinalLoss = nn.CrossEntropyLoss(reduction='sum')

    cudnn.benchmark = True

    NUM_ITER = max(len(source_train_dataset_main), len(target_train_dataset_main)) // batch_size

    for epoch in range(epoch0, num_epochs):  # loop over the dataset multiple times
        print(f'Epoch {epoch + 1}/{num_epochs}')
        since = time.time()
        running_loss_m = 0.0

        source_iter = loopy(source_train_main_dataloader)
        target_iter = loopy(target_train_main_dataloader)

        net.train()

        for it in range(NUM_ITER):

            rimgs, dimgs, labels = next(source_iter)

            # set to train and zero the parameter gradients
            optimizer.zero_grad()

            # Bring data over the device of choice
            rimgs = rimgs.to(device)
            dimgs = dimgs.to(device)
            labels = labels.to(device)

            # forward
            s_fc2_emb, outputs = net(rimgs, dimgs)
            # compute main loss
            s_cls_loss = criterion(outputs, labels)

            s_fc2_ring_loss = weight_L2norm * get_L2norm_loss_self_driven(s_fc2_emb, radius)

            rimgs, dimgs, _ = next(target_iter)

            # Bring data over the device of choice
            rimgs = rimgs.to(device)
            dimgs = dimgs.to(device)

            # forward
            t_fc2_emb, _ = net(rimgs, dimgs)

            t_fc2_ring_loss = weight_L2norm * get_L2norm_loss_self_driven(t_fc2_emb, radius)

            loss = s_cls_loss + s_fc2_ring_loss + t_fc2_ring_loss
            loss.backward()

            # update weights
            optimizer.step()

            # print statistics
            running_loss_m += s_cls_loss.item()
            if it % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {it + 1}] Lm {running_loss_m / 100}')
                running_loss_m = 0.

            it += 1

        net.train(False)
        # ************************
        # SOURCE VALIDATION
        # ************************
        source_loss = 0
        running_corrects = 0
        for images_rgb, images_d, labels in source_test_main_dataloader:
            images_rgb = images_rgb.to(device)
            images_d = images_d.to(device)
            labels = labels.to(device)

            _, outputs = net(images_rgb, images_d)
            loss = criterionFinalLoss(outputs, labels)
            source_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).data.item()

        source_loss = source_loss / float(len(source_test_dataset_main))
        source_losses.append(source_loss)
        source_acc = running_corrects / float(len(source_test_dataset_main))
        source_accs.append(source_acc)

        # ************************
        # TARGET VALIDATION
        # ************************
        target_loss = 0
        running_corrects = 0
        for images_rgb, images_d, labels in target_validation_dataloader:
            images_rgb = images_rgb.to(device)
            images_d = images_d.to(device)
            labels = labels.to(device)

            _, outputs = net(images_rgb, images_d)
            loss = criterionFinalLoss(outputs, labels)
            target_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).data.item()

        target_loss = target_loss / float(len(target_test_dataset_main))
        target_losses.append(target_loss)
        target_acc = running_corrects / float(len(target_test_dataset_main))
        target_accs.append(target_acc)

        scheduler.step()

        # CHECKPOINT
        if checkpoint_dir is not None:
            filename = str(epoch + 1) + '.ckpt'
            path = os.path.join(checkpoint_dir, filename)
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

        time_elapsed = time.time() - since
        print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return source_losses, target_losses, source_accs, target_accs


def train_sourceonly_singlemod_HAFN(net, modality,
                                    source_train_dataset_main,
                                    target_train_dataset_main,
                                    source_test_dataset_main,
                                    target_test_dataset_main,
                                    batch_size, lr, momentum, step_size, gamma, num_epochs, checkpoint_dir,
                                    weight_decay,
                                    radius, weight_L2norm, dropout_p):
    """
  modality = RGB / depth
  """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    # Load checkpoint if available
    checkpoint = None
    if checkpoint_dir is not None:
        checkpoint = load_checkpoint(checkpoint_dir)
    if checkpoint is not None:
        net.load_state_dict(checkpoint['net'])

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if checkpoint is not None:
        epoch0 = checkpoint['epoch'] + 1  # starting from the next epoch
        source_losses = checkpoint['source_losses']
        source_accs = checkpoint['source_accs']
        target_losses = checkpoint['target_losses']
        target_accs = checkpoint['target_accs']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        print(f'Checkpoint found! Starting from epoch {epoch0}')

    else:
        epoch0 = 0
        source_losses = []
        source_accs = []
        target_losses = []
        target_accs = []

        print(f'No checkpoint found, starting from epoch 1')

    source_train_main_dataloader = DataLoader(source_train_dataset_main, batch_size=batch_size, shuffle=True,
                                              num_workers=1, drop_last=True)

    # TARGET TRAIN
    target_train_main_dataloader = DataLoader(target_train_dataset_main, batch_size=batch_size, shuffle=True,
                                              num_workers=1, drop_last=True)

    # TARGET
    target_validation_dataloader = DataLoader(target_test_dataset_main, batch_size=batch_size, shuffle=True,
                                              num_workers=1,
                                              drop_last=False)

    # used in validation (drop_last = False)
    source_test_main_dataloader = DataLoader(source_test_dataset_main, batch_size=batch_size, shuffle=True,
                                             num_workers=1,
                                             drop_last=False)

    criterion = nn.CrossEntropyLoss()
    criterionFinalLoss = nn.CrossEntropyLoss(reduction='sum')

    cudnn.benchmark =True

    NUM_ITER = max(len(source_train_dataset_main), len(target_train_dataset_main)) // batch_size

    for epoch in range(epoch0, num_epochs):
        print('Starting epoch {}/{}'.format(epoch + 1, num_epochs))
        # ************************
        # TRAINING
        # ************************
        net.train()
        source_iter = loopy(source_train_main_dataloader)
        target_iter = loopy(target_train_main_dataloader)

        since = time.time()
        for it in range(NUM_ITER):

            rimgs, dimgs, labels = next(source_iter)

            optimizer.zero_grad()

            if modality == 'RGB':
                images = rimgs
            else:
                images = dimgs

            images = images.to(device)
            labels = labels.to(device)

            if modality == 'RGB':
                s_fc2_emb, outputs = net(images, None)

            else:
                s_fc2_emb, outputs = net(None, images)

            # compute main loss
            s_cls_loss = criterion(outputs, labels)

            s_fc2_ring_loss = weight_L2norm * get_L2norm_loss_self_driven(s_fc2_emb, radius)

            rimgs, dimgs, _ = next(target_iter)

            if modality == 'RGB':
                images = rimgs
            else:
                images = dimgs

            images = images.to(device)

            if modality == 'RGB':
                t_fc2_emb, _ = net(images, None)

            else:
                t_fc2_emb, _ = net(None, images)

            t_fc2_ring_loss = weight_L2norm * get_L2norm_loss_self_driven(t_fc2_emb, radius)

            loss = s_cls_loss + s_fc2_ring_loss + t_fc2_ring_loss

            loss.backward()
            optimizer.step()  # update weights

        net.train(False)

        # ************************
        # SOURCE VALIDATION
        # ************************
        source_loss = 0
        running_corrects = 0
        for images_rgb, images_d, labels in source_test_main_dataloader:
            if modality == 'RGB':
                images = images_rgb
            else:
                images = images_d

            images = images.to(device)
            labels = labels.to(device)

            if modality == 'RGB':
                _, outputs = net(images, None)

            else:
                _, outputs = net(None, images)

            loss = criterionFinalLoss(outputs, labels)
            source_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).data.item()

        source_loss = source_loss / float(len(source_test_dataset_main))
        source_losses.append(source_loss)
        source_acc = running_corrects / float(len(source_test_dataset_main))
        source_accs.append(source_acc)

        # ************************
        # TARGET VALIDATION
        # ************************
        target_loss = 0
        running_corrects = 0
        for images_rgb, images_d, labels in target_validation_dataloader:
            if modality == 'RGB':
                images = images_rgb
            else:
                images = images_d

            images = images.to(device)
            labels = labels.to(device)

            if modality == 'RGB':
                _, outputs = net(images, None)

            else:
                _, outputs = net(None, images)

            loss = criterionFinalLoss(outputs, labels)
            target_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).data.item()

        target_loss = target_loss / float(len(target_test_dataset_main))
        target_losses.append(target_loss)
        target_acc = running_corrects / float(len(target_test_dataset_main))
        target_accs.append(target_acc)

        scheduler.step()

        # CHECKPOINT
        if checkpoint_dir is not None:
            filename = str(epoch + 1) + '.ckpt'
            path = os.path.join(checkpoint_dir, filename)
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

        time_elapsed = time.time() - since
        print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # delete checkpoints
    # files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    # for f in files:
    #    os.remove(f)
    # print('Finished Training. Checkpoints deleted.')
    return source_losses, target_losses, source_accs, target_accs
