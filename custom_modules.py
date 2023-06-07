import torch.nn as nn
import torch

class MultilayerPerceptron(nn.module):

    def __init__(self, input_size, hidden_size):
        super(MultilayerPerceptron, self).__init__()
        
        # svaing the initialization parameters
        self.input_size = input_size
        self.hidden_size = hidden_size

        # defining our model
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear2(self.input_size, self.hidden_size),
            nn.Sigmoid()

        )

        def forward(self, x):
            output = self.model(x)
            return output

class MultilayerPerceptrontwo(nn.module):
    def __init__(self, input_size, hidden_size):
        super(MultilayerPerceptrontwo, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # define our layers
        self.linear = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size, self.input_size)
        self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            linear = self.linear(x)
            relu = self.relu(linear)
            linear2 = self.linear2(relu)
            output = self.sigmoid(linear2)
            return output

# Make a sample input
input = torch.randn(2, 5)

# Create our model
model = MultilayerPerceptron(5, 3)

# Pass our input through our model
model(input)

import torch.optim as optim

x = torch.randn(2,5)
y = torch.randn(2,3)

model = MultilayerPerceptron(5, 3)
loss_function = nn.BCELoss()
y_pred = model(x)
loss_function(y_pred, y).item
  

n_epoch = 10
adam = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(n_epoch):
    adam.zero_grad()

    y_pred = model(x)
    loss = loss_function(y_pred, y)
    print(f"Epoch {epoch} | Loss: {loss}")
    
    # compute the gradients 

    loss.backward()
    adam.step()