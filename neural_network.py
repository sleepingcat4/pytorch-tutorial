import torch.nn as nn
import torch

# create the inputs
input = torch.ones(2, 3, 4)

# Make a linear layers trasnforming N, H_in dimensinal inputs to N, *H_out
linear = nn.Linear(4, 5)
linear_output = linear(input)
print(linear_output)

