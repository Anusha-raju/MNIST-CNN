from __future__ import print_function
from tqdm import tqdm
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

cuda = torch.cuda.is_available()
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
device = torch.device("cuda" if cuda else "cpu")
print(device)

class FitEvaluate:
  def __init__(self,model,device, train_loader,test_loader):
    self.model, self.device, self.train_loader, self.test_loader = model, device, train_loader,test_loader
    self.train_losses,self.test_losses, self.train_acc, self.test_acc= [], [], [], []
    pass

  def train(self, optimizer, epoch):
    try:
      self.model.train()
      pbar = tqdm(self.train_loader)
      correct = 0
      processed = 0
      for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = self.model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        self.train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        self.train_acc.append(100*correct/processed)

    except Exception as ex:
      print(f"Exception in train function: {ex}")

  def test(self):
    try:
      self.model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in self.test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(self.test_loader.dataset)
      self.test_losses.append(test_loss)

      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, len(self.test_loader.dataset),
          100. * correct / len(self.test_loader.dataset)))

      self.test_acc.append(100. * correct / len(self.test_loader.dataset))

    except Exception as ex:
        print(f"Exception in test function: {ex}")

  def plot_accuracy_loss(self):
    t = [t_items.item() for t_items in self.train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(self.train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(self.test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(self.test_acc)
    axs[1, 1].set_title("Test Accuracy")

  def epoch_training(self,optimizer,scheduler = '',EPOCHS=15):
    try:
      for epoch in range(EPOCHS):
          print("EPOCH:", epoch)
          self.train(optimizer, epoch)
          if scheduler:
            scheduler.step()
          self.test()
      self.plot_accuracy_loss()

    except Exception as ex:
      print(f"Exception in test function: {ex}")


class MNIST_DATA:
  def __init__(self, train_transforms,test_transforms):
    self.train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    self.test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    self.train_loader = torch.utils.data.DataLoader(self.train, **dataloader_args) # train dataloader
    self.test_loader = torch.utils.data.DataLoader(self.test, **dataloader_args) # test dataloader
    print("Loaded the MNIST data!!!")
    pass

  def stats(self):
    try:
      train_data = self.train.train_data
      train_data = self.train.transform(train_data.numpy())
      print("********* Train Data Stats *********")
      print(' - Numpy Shape:', self.train.train_data.cpu().numpy().shape)
      print(' - Tensor Shape:', self.train.train_data.size())
      print(' - min:', torch.min(train_data))
      print(' - max:', torch.max(train_data))
      print(' - mean:', torch.mean(train_data))
      print(' - std:', torch.std(train_data))
      print(' - var:', torch.var(train_data))

    except Exception as ex:
      print(f"Exception in stats function: {ex}")


  def showimages(self, num_of_images):
    try:
        train_loader = self.train_loader
        dataiter = iter(train_loader)
        images, labels = next(dataiter)  # Use `next(dataiter)` instead of `dataiter.next()`

        print(" Batch Shape: ", images.shape)
        print(" Labels of the images: ", labels.shape)
        print(" Images")
        figure = plt.figure()
        for index in range(1, num_of_images + 1):
            plt.subplot(6, 10, index)
            plt.axis('off')
            plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        plt.show()  # Make sure to display the plot

    except Exception as ex:
        print(f"Exception in showimages function: {ex}")