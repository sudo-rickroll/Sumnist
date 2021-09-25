import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class Process:
    def __init__(self, num_of_epochs, validate_per_epoch, mode, model, device, loss_func, train_loader, optimiser, test_loader, save_checkpoint):
        self.epochs = num_of_epochs
        self.validation_freq = validate_per_epoch
        self.mode = mode
        self.model = model
        self.device = device
        self.loss_func = getattr(F, loss_func)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimiser = optimiser
        self.checkpoint = save_checkpoint
        self.accs_mnist, self.accs_total, self.losses_mnist, self.losses_total = {'train':[], 'test':[] }, {'train':[], 'test':[] }, {'train':[], 'test':[] }, {'train':[], 'test':[] }

    def train(self):
      accuracy_value_mnist, accuracy_value_total, loss_value_mnist, loss_value_total=0,0,0,0
      for images, labels, rands, total in self.train_loader:
        images, labels, rands, total = images.to(self.device), labels.to(self.device), rands.to(self.device), total.to(self.device)
        pred_mnist, pred_sum = self.model(images, rands)
        loss = self.loss_func(pred_mnist, labels) + self.loss_func(pred_sum, total)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        accuracy_value_mnist += pred_mnist.argmax(dim=1).eq(labels.view_as(pred_mnist.argmax(dim=1))).sum().item()
        accuracy_value_total += pred_sum.argmax(dim=1).eq(total.view_as(pred_sum.argmax(dim=1))).sum().item()
        loss_value_total += loss.item()
        loss_value_mnist += self.loss_func(pred_mnist, labels).item()
      return (accuracy_value_mnist/(len(self.train_loader) * self.train_loader.batch_size)) * 100, (accuracy_value_total/(len(self.train_loader) * self.train_loader.batch_size)) * 100, loss_value_mnist, loss_value_total

    def test(self, show_pred=False):
      accuracy_value_mnist, accuracy_value_total, loss_value_mnist, loss_value_total=0,0,0,0
      if show_pred:
            random_batch = random.randint(0, len(self.test_loader) -1)
            random_sample = random.randint(0, self.test_loader.batch_size - 1)
            image, label, rand, total, pred_mnist_value, pred_sum_value = 0, 0, 0, 0, 0, 0
      with torch.no_grad():
        for index, batch in enumerate(self.test_loader):
          images, labels, rands, totals = batch
          images, labels, rands, totals = images.to(self.device), labels.to(self.device), rands.to(self.device), totals.to(self.device)
          pred_mnist, pred_sum = self.model(images, rands)
          loss = self.loss_func(pred_mnist, labels) + self.loss_func(pred_sum, totals)
          pred_mnist_values, pred_sum_values = pred_mnist.argmax(dim=1), pred_sum.argmax(dim=1)
          accuracy_value_mnist += pred_mnist_values.eq(labels.view_as(pred_mnist_values)).sum().item()
          accuracy_value_total += pred_sum_values.eq(totals.view_as(pred_sum_values)).sum().item()
          loss_value_total += loss.item()
          loss_value_mnist += self.loss_func(pred_mnist, labels).item()
          if show_pred and index == random_batch:
            image, label, rand, total = images[random_sample], labels[random_sample].item(), rands[random_sample].item(), totals[random_sample].item()
            pred_mnist_value, pred_sum_value = pred_mnist_values[random_sample].item(), pred_sum_values[random_sample].item()
        if show_pred:
          print('\nSample of validation : \n')
          plt.subplot(1,5,1)
          plt.imshow(image.squeeze().cpu(), cmap='gray')
          plt.axis('off')
          plt.title(f'MNIST Image\nActual Value : {label}\nPredicted Value : {pred_mnist_value}')
          plt.subplot(1,5,2)
          plt.text(0.5, 0.5, "+", fontsize=14)
          plt.axis('off')
          plt.subplot(1,5,3)
          plt.text(0.5, 0.5, rand, fontsize=21)
          plt.axis('off')
          plt.text(-0.2, 0.7, 'Random Digit', fontsize=13)
          plt.subplot(1,5,4)
          plt.text(0.5, 0.5, "=", fontsize=14)
          plt.axis('off')
          plt.subplot(1,5,5)
          plt.text(0.5, 0.5, pred_sum_value, fontsize=21)
          plt.axis('off')
          plt.text(-0.2, 0.7, f'Actual Total : {total}\nPredicted Total : {pred_sum_value}', fontsize=13)
          plt.savefig(f'./images/Sample Validation.jpg', bbox_inches='tight')
          plt.show()
          print("Image stored in the 'images' folder")
      return (accuracy_value_mnist/(len(self.test_loader) * self.test_loader.batch_size)) * 100, (accuracy_value_total/(len(self.test_loader) * self.test_loader.batch_size)) * 100, loss_value_mnist, loss_value_total

    def run(self): 
      if 'train' in self.mode:
        print('Training the model...')     
        for epoch in range(self.epochs):        
          acc_mnist, acc_total, loss_mnist, loss_total = self.train()
          if (epoch + 1) % self.validation_freq == 0:
              print(f'\nEpoch {epoch + 1} : \n\nTraining stats : \nAccuracy MNIST= {acc_mnist : .2f} \nAccuracy Total= {acc_total : .2f} \nLoss MNIST= {loss_mnist : .8f} \nLoss Total= {loss_total : .8f}\n')
          self.accs_mnist['train'] += [acc_mnist]
          self.accs_total['train'] += [acc_total]
          self.losses_mnist['train'] += [loss_mnist]
          self.losses_total['train'] += [loss_total]
          acc_mnist, acc_total, loss_mnist, loss_total = self.test()
          if (epoch + 1) % self.validation_freq == 0:
              print(f'Validation stats : \nAccuracy MNIST= {acc_mnist : .2f} \nAccuracy Total= {acc_total : .2f} \nLoss MNIST= {loss_mnist : .8f} \nLoss Total= {loss_total : .8f}\n------------------------')
          self.accs_mnist['test'] += [acc_mnist]
          self.accs_total['test'] += [acc_total]
          self.losses_mnist['test'] += [loss_mnist]
          self.losses_total['test'] += [loss_total]
        if self.checkpoint:
          print(f"Saving model checkpoint in '{self.checkpoint}'")
          try:
            checkpoint_pth = {'model' : self.model.state_dict()}
            torch.save(checkpoint_pth, self.checkpoint)
            print('Model saved successfully')
          except:
            print('Model could not be saved')
      if 'validate' in self.mode:
        print('\nValidating the model...')
        acc_mnist, acc_total, loss_mnist, loss_total = self.test(show_pred=True)
        print(f'\nValidation stats : \nAccuracy MNIST= {acc_mnist : .2f} \nAccuracy Total= {acc_total : .2f} \nLoss MNIST= {loss_mnist : .8f} \nLoss Total= {loss_total : .8f}\n------------------------')
          
      




  
