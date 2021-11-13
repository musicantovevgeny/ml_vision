## Code by Evgeny Musicantov
## Following
## https://www.youtube.com/watch?v=c36lUUr864M - Python Engineer
## https://www.youtube.com/watch?v=Jy4wM2X21u0 - Alladin Persson

# Imports
import numpy as np
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
import math



class simple_2l_model:
    def __init__(self, input_size, num_classes, hn):
        self.w1 = torch.empty((input_size,hn), requires_grad=True)
        self.b1 = torch.zeros(hn, requires_grad=True)
        self.w2 = torch.empty((hn,num_classes), requires_grad=True)
        self.b2 = torch.zeros(num_classes, requires_grad=True)
        with torch.no_grad():
            self.w1.normal_(0, 1./ math.sqrt(hn)) 
            self.w2.normal_(0, 1./ math.sqrt(num_classes)) 
        
    def __call__(self,x):
        with torch.no_grad():
            return self._feed_forward(x)

    def _feed_forward(self, x):
        x = torch.matmul(x,self.w1)+ self.b1
        x = F.relu(x)
        x = torch.matmul(x, self.w2) + self.b2
        return x


    def forward(self, x):
        return self._feed_forward(x)

class nn_model(nn.Module):
    def __init__(self, input_size, num_classesm, hn=50):
        super(nn_model, self).__init__()
        self.fc1 = nn.Linear(input_size, hn)
        self.fc2 = nn.Linear(hn, num_classes)

    def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

# Hyperparameters of our neural network which depends on the dataset, and
# also just experimenting to see what works well (learning rate for example).
input_size = 784
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 10
num_of_hidden_layers = 150

# Load Training and Test data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = simple_2l_model(input_size, num_classes, hn= num_of_hidden_layers)

# Train Network
for epoch in range(num_epochs):
    epoch_loss = 0
    num_of_batches = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # data is grayscaler image 0<= <=1
        data = data.reshape(data.shape[0], -1)
        y = F.one_hot(targets,num_classes=num_classes)
        
        logits = model.forward(data)
        y_hat = F.softmax(logits, dim=1)
        loss = - (y * torch.log(y_hat)).mean()
        epoch_loss += loss.item() 
        num_of_batches += 1
        

        # calculate the grads
        loss.backward()

        # update weight - like optimizer stem
        with torch.no_grad():
            model.w1 -= learning_rate*model.w1.grad
            model.w2 -= learning_rate*model.w2.grad
            model.b1 -= learning_rate*model.b1.grad
            model.b2 -= learning_rate*model.b2.grad

        # zero the grads for further calc 
        model.w1.grad.zero_()
        model.b1.grad.zero_()
        model.w2.grad.zero_()
        model.b2.grad.zero_()
    epoch_loss = epoch_loss/num_of_batches
    print(f'{epoch = } , {epoch_loss = :.4f}')


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)

            logits = model(x)
            _, predictions = logits.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return float(num_correct)/num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
