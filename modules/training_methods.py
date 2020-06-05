import torch
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import time


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


def train_RGBD_DA(net,
                  source_train_dataset_main, source_train_dataset_pretext,
                  target_dataset_main, target_dataset_pretext,
                  source_test_dataset_main, source_test_dataset_pretext,
                  batch_size, num_epochs, lr, momentum, step_size, gamma, entropy_weight, lamda):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # Losses and accuracies on the main task
  source_losses = []
  source_accs = []
  target_losses = []
  target_accs = []

  # Data loaders for training phase
  # SOURCE
  source_train_main_dataloader = DataLoader(source_train_dataset_main, batch_size=batch_size, shuffle=True,
                                            num_workers=4, drop_last=True)
  source_train_pretext_dataloader = DataLoader(source_train_dataset_pretext, batch_size=batch_size, shuffle=True,
                                               num_workers=4, drop_last=True)
  # TARGET
  target_main_dataloader = DataLoader(target_dataset_main, batch_size=batch_size, shuffle=True, num_workers=4,
                                      drop_last=True)
  target_pretext_dataloader = DataLoader(target_dataset_pretext, batch_size=batch_size, shuffle=True, num_workers=4,
                                         drop_last=True)

  target_validation_dataloader = DataLoader(target_dataset_main, batch_size=batch_size, shuffle=True, num_workers=4,
                                         drop_last=False)

  # used in validation (drop_last = False)
  source_test_main_dataloader = DataLoader(source_test_dataset_main, batch_size=batch_size, shuffle=True, num_workers=4,
                                           drop_last=False)
  source_test_pretext_dataloader = DataLoader(source_test_dataset_pretext, batch_size=batch_size, shuffle=True,
                                              num_workers=4, drop_last=False)

  criterion = nn.CrossEntropyLoss()
  criterionFinalLoss = nn.CrossEntropyLoss(reduction='sum')

  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

  net = net.to(device)
  cudnn.benchmark

  NUM_ITER = max(len(source_train_dataset_main), len(target_dataset_main)) // batch_size

  for epoch in range(num_epochs):  # loop over the dataset multiple times
    print(f'Epoch {epoch+1}/{num_epochs}')
    since = time.time()
    running_loss_m = 0.0
    running_loss_p = 0.0
    running_entropy = 0.0

    source_data_main_iter = loopy(source_train_main_dataloader)
    target_data_main_iter = loopy(target_main_dataloader)
    source_data_pretext_iter = loopy(source_train_pretext_dataloader)
    target_data_pretext_iter = loopy(target_pretext_dataloader)

    for it in range(NUM_ITER):

      # set to train and zero the parameter gradients
      net.train()
      optimizer.zero_grad()

      # ************************
      # SOURCE MAIN FORWARD PASS
      # ************************
      # unpack in RGB images, depth images and labels
      rimgs, dimgs, labels = next(source_train_main_dataloader)

      # Bring data over the device of choice
      rimgs = rimgs.to(device)
      dimgs = dimgs.to(device)
      labels = labels.to(device)

      # forward
      outputs = net(rimgs, dimgs)
      # compute main loss
      loss_m = criterion(outputs, labels)

      # ******************************************
      # TARGET MAIN FORWARD PASS WITH ENTROPY LOSS
      # ******************************************

      rimgt, dimgt, _ = next(target_main_dataloader)

      rimgt = rimgt.to(device)
      dimgt = dimgt.to(device)

      outputs = net(rimgt, dimgt)

      new_loss_m = loss_m + entropy_weight * entropy_loss(outputs)
      running_entropy += entropy_weight * entropy_loss(outputs)

      new_loss_m.backward()

      # ***************************
      # SOURCE PRETEXT FORWARD PASS
      # ***************************
      # using same batch as main forward pas

      rimgs, dimgs, labels = next(source_train_pretext_dataloader)

      rimgs = rimgs.to(device)
      dimgs = dimgs.to(device)
      labels = labels.to(device)

      outputs = net(rimgs, dimgs, lamda)
      loss_sp = criterion(outputs, labels)
      loss_sp.backward()

      # ***************************
      # TARGET PRETEXT FORWARD PASS
      # ***************************

      rimgt, dimgt, labels = next(target_pretext_dataloader)

      rimgt = rimgt.to(device)
      dimgt = dimgt.to(device)
      labels = labels.to(device)

      outputs = net(rimgt, dimgt, lamda)

      loss_tp = criterion(outputs, labels)
      # old: new_loss_tp = loss_tp + ENTROPY_WEIGHT * entropy_loss(outputs)
      loss_tp.backward()

      # update weights
      optimizer.step()

      # print statistics
      running_loss_m += loss_m.item()
      running_loss_p += (loss_sp+loss_tp).item()
      if it % 100 == 99:    # print every 100 mini-batches
        print(f'[{epoch+1}, {it+1}] Lm {running_loss_m/100}, Lp {running_loss_p/100}, EntropyLoss {running_entropy/100}')
        running_loss_m = 0.
        running_loss_p = 0.
        running_entropy = 0.


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

      outputs = net(images_rgb, images_d)
      loss = criterionFinalLoss(outputs,labels)
      source_loss += loss.item()

      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
    
    source_loss = source_loss/float(len(source_test_dataset_main))
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

      outputs = net(images_rgb, images_d)
      loss = criterionFinalLoss(outputs,labels)
      target_loss += loss.item()

      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
    
    target_loss = target_loss/float(len(target_dataset_main))
    target_losses.append(target_loss)
    target_acc = running_corrects / float(len(target_dataset_main))
    target_accs.append(target_acc)


    scheduler.step()
    

    # CHECKPOINT
    filename = 'checkpoint_end_epoch'+str(epoch+1)
    path = f"/content/drive/My Drive/{filename}"
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
    """
    Example to load:
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    """
    # TODO add load part and remove old checkpoints
    time_elapsed = time.time() - since
    print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

  print('Finished Training')

def RGBD_e2e(net,
             source_train_dataset_main,
             target_dataset_main,
             source_test_dataset_main,
             batch_size, num_epochs, lr, momentum, step_size, gamma):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # Losses and accuracies on the main task
  source_losses = []
  source_accs = []
  target_losses = []
  target_accs = []

  # Data loaders for training phase
  # SOURCE
  source_train_main_dataloader = DataLoader(source_train_dataset_main, batch_size=batch_size, shuffle=True,
                                            num_workers=4, drop_last=True)

  # TARGET
  target_validation_dataloader = DataLoader(target_dataset_main, batch_size=batch_size, shuffle=True, num_workers=4,
                                         drop_last=False)

  # used in validation (drop_last = False)
  source_test_main_dataloader = DataLoader(source_test_dataset_main, batch_size=batch_size, shuffle=True, num_workers=4,
                                           drop_last=False)

  criterion = nn.CrossEntropyLoss()
  criterionFinalLoss = nn.CrossEntropyLoss(reduction='sum')

  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

  net = net.to(device)
  cudnn.benchmark

  for epoch in range(num_epochs):  # loop over the dataset multiple times
    print(f'Epoch {epoch+1}/{num_epochs}')
    since = time.time()
    running_loss_m = 0.0
    running_loss_p = 0.0
    running_entropy = 0.0

    for rimgs, dimgs, labels in source_train_main_dataloader:

      # set to train and zero the parameter gradients
      net.train()
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
      if it % 100 == 99:    # print every 100 mini-batches
        print(f'[{epoch+1}, {it+1}] Lm {running_loss_m/100}')
        running_loss_m = 0.


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

      outputs = net(images_rgb, images_d)
      loss = criterionFinalLoss(outputs,labels)
      source_loss += loss.item()

      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
    
    source_loss = source_loss/float(len(source_test_dataset_main))
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

      outputs = net(images_rgb, images_d)
      loss = criterionFinalLoss(outputs,labels)
      target_loss += loss.item()

      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
    
    target_loss = target_loss/float(len(target_dataset_main))
    target_losses.append(target_loss)
    target_acc = running_corrects / float(len(target_dataset_main))
    target_accs.append(target_acc)


    scheduler.step()
    

    # CHECKPOINT
    filename = 'checkpoint_end_epoch'+str(epoch+1)
    path = f"/content/drive/My Drive/{filename}"
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
    """
    Example to load:
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    """
    # TODO add load part and remove old checkpoints
    time_elapsed = time.time() - since
    print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

  print('Finished Training')

def train_sourceonly_singlemod(net, modality,
                               source_train_dataset, source_test_dataset,
                               target_dataset, batch_size, lr, momentum, step_size, gamma, num_epochs):
  """
  modality = RGB / depth
  """
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

  source_losses = []
  source_accs = []
  target_losses = []
  target_accs = []

  source_train_dataloader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

  # used in validation (drop_last = False)
  source_test_dataloader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
  target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

  criterion = nn.CrossEntropyLoss()
  criterionFinalLoss = nn.CrossEntropyLoss(reduction='sum')

  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

  net = net.to(DEVICE) 
  cudnn.benchmark 

  for epoch in range(num_epochs):
    print('Starting epoch {}/{}'.format(epoch+1, num_epochs))
    # ************************
    # TRAINING
    # ************************
    net.train()

    since = time.time()
    for images_rgb, images_d, labels in source_train_dataloader:
      optimizer.zero_grad()
      if modality == 'RGB':
        images = images_rgb
      else:
        images = images_d
      
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      outputs = net(images)

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step() # update weights 
    
    net.train(False)

    # ************************
    # SOURCE VALIDATION
    # ************************
    source_loss = 0 
    running_corrects = 0
    for images_rgb, images_d, labels in source_test_dataloader:
      if modality == 'RGB':
        images = images_rgb
      else:
        images = images_d
      
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      outputs = net(images)
      loss = criterionFinalLoss(outputs,labels)
      source_loss += loss.item()

      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
    
    source_loss = source_loss/float(len(source_test_dataset))
    source_losses.append(source_loss)
    source_acc = running_corrects / float(len(source_test_dataset))
    source_accs.append(source_acc)


    # ************************
    # TARGET VALIDATION
    # ************************
    target_loss = 0 
    running_corrects = 0
    for images_rgb, images_d, labels in target_dataloader:
      if modality == 'RGB':
        images = images_rgb
      else:
        images = images_d
      
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      outputs = net(images)
      loss = criterionFinalLoss(outputs,labels)
      target_loss += loss.item()

      _, preds = torch.max(outputs.data, 1)
      running_corrects += torch.sum(preds == labels.data).data.item()
    
    target_loss = target_loss/float(len(target_dataset))
    target_losses.append(target_loss)
    target_acc = running_corrects / float(len(target_dataset))
    target_accs.append(target_acc)


    scheduler.step()

    # CHECKPOINT
    filename = modality+'_checkpoint_end_epoch'+str(epoch+1)
    path = F"/content/drive/My Drive/{filename}" 
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
    """
    Example to load:
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    """


    time_elapsed = time.time() - since
    print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
