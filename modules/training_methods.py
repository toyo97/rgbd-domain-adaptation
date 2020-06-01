from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim

def entropy_loss(logits):
    p_softmax = F.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to, used for numerical stability
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))

def RGBD_DA(net, source_train_dataset, source_test_dataset, target_dataset):
  # TODO: insert hyperparameters
  # Data loaders for synROD - MAIN/PRETEXT task only at training
  source_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

  # Data loader for ROD train and test - PRETEXT at train, MAIN at test (check validity of drop last when testing)
  target_dataloader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

  # Define loss
  # Both main and pretext losses are computed with the cross entropy function
  criterion = nn.CrossEntropyLoss()

  # Define optimizer
  # TODO try with different optimizers for the three components of the network
  optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

  # Define scheduler
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

  NUM_ITER = max(len(source_train_dataset), len(target_dataset)) // BATCH_SIZE

  # Allow iterating over a dataset more than once
  # to deal with different number of samples between datasets
  # during training and batch sampling
  def loopy(dl):
    while True:
      for x in dl: yield x

  # By default, everything is loaded to cpu
  net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

  cudnn.benchmark # optimizes runtime

  for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    since = time.time()
    running_loss_m = 0.0
    running_loss_p = 0.0

    source_data_iter = loopy(source_dataloader)
    target_data_iter = loopy(target_dataloader)

    for it in range(NUM_ITER):

      # set to train and zero the parameter gradients
      net.train()
      optimizer.zero_grad()

      # ************************
      # SOURCE MAIN FORWARD PASS
      # ************************
      # unpack in RGB images, depth images and labels
      rimgs, dimgs, labels = next(source_data_iter)

      # Bring data over the device of choice
      rimgs = rimgs.to(DEVICE)
      dimgs = dimgs.to(DEVICE)
      labels = labels.to(DEVICE)

      # forward
      outputs = net(rimgs, dimgs)
      # compute main loss
      loss_m = criterion(outputs, labels)

      # ***************************
      # TARGET MAIN FORWARD PASS
      # ***************************

      rimgt, dimgt, _ = next(target_data_iter)

      rimgt = rimgt.to(DEVICE)
      dimgt = dimgt.to(DEVICE)

      outputs = net(rimgt, dimgt)

      new_loss_m = loss_m + ENTROPY_WEIGHT * entropy_loss(outputs)

      new_loss_m.backward()

      # ***************************
      # SOURCE PRETEXT FORWARD PASS
      # ***************************
      # using same batch as main forward pas

      rimgs, dimgs, labels = transform_batch(rimgs, dimgs)

      rimgs = rimgs.to(DEVICE)
      dimgs = dimgs.to(DEVICE)
      labels = labels.to(DEVICE)

      outputs = net(rimgs, dimgs, LAMBDA)
      loss_sp = criterion(outputs, labels)
      loss_sp.backward()

      # ***************************
      # TARGET PRETEXT FORWARD PASS
      # ***************************

      rimgt, dimgt, labels = transform_batch(rimgt, dimgt)

      rimgt = rimgt.to(DEVICE)
      dimgt = dimgt.to(DEVICE)
      labels = labels.to(DEVICE)

      outputs = net(rimgt, dimgt, LAMBDA)

      loss_tp = criterion(outputs, labels)
      #new_loss_tp = loss_tp + ENTROPY_WEIGHT * entropy_loss(outputs)
      loss_tp.backward()

      # update weights
      optimizer.step()

      # print statistics
      running_loss_m += loss_m.item()
      running_loss_p += (loss_sp+loss_tp).item()
      if it % 100 == 99:    # print every 100 mini-batches
        print(f'[{epoch+1}, {it+1}] Lm {running_loss_m/100}, Lp {running_loss_p/100}')
        running_loss_m = 0.
        running_loss_p = 0.

      # TODO: validation

    scheduler.step()
    time_elapsed = time.time() - since
    print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

  print('Finished Training')

def RGBD_e2e():
  

def train_sourceonly_singlemod(net, modality, source_train_dataset, source_test_dataset, target_dataset, batch_size, lr, momentum, step_size, gamma, num_epochs):
  """
  modality = RGB / depth
  """
  # TO THINK:
  # get_item of the dataset returns both modalities -> uses two different apposite Classes of Dataset?
  # after each epoch I validate on the whole ROD or on a part of it? (besides validating on source)

  source_losses = []
  source_accs = []
  target_losses = []
  target_accs = []

  source_train_dataloader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
  criterion = nn.CrossEntropyLoss()

  # used in validation (drop_last = False)
  source_test_dataloader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
  target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
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
    
    target_loss = target_loss/float(len(source_test_dataset))
    target_losses.append(target_loss)
    target_acc = running_corrects / float(len(source_test_dataset))
    target_accs.append(target_acc)


    scheduler.step()
    time_elapsed = time.time() - since
    print('Time to complete the epoch: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

