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
  
  
def RGBD_sourceonly(split=RGB/depth):

def RGBD_e2e():
