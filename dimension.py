import torch

x = torch.zeros(3,2)

# Add a new dimension of size 1 at the 1st dimension
x = x.unsqueeze(1)

# Squeeze the dimensions of x by getting rid of all the dimensions with 1 element
x = x.squeeze()

# to get the total number of elements in a tensor
x

# Get the number of elements in tensor.
x.numel()

# Initialize an example tensor
x = torch.Tensor([[1, 2], [3, 4]])

# checking if cuda is available
if torch.cuda.is_available():
    x.to('cuda')

# check device
device = x.device