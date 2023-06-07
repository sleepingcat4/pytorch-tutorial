import torch

# Initialize a base tensor
x = torch.tensor([[1., 2.], [3., 4.]])

# Initialize a tensor of 0s
x_zeros = torch.zeros_like(x)

# Initialize a tensor of 1s
x_ones = torch.ones_like(x)

# Initialize a tensor where each element is sampled from a uniform distribution
# between 0 and 1
x_rand = torch.rand_like(x)

# we can create tensor specifying the shape
# Initialize a 2x3x2 tensor of 0s
shape = (4, 2, 2)
x_zeros = torch.zeros(shape) # x_zeros = torch.zeros(4, 3, 2) is an alternative

# Create a tensor with values 0-9
x = torch.arange(10)

x = torch.ones(3,2)

# Initialize a 3x2 tensor, with 3 rows and 2 columns
x = torch.Tensor([[1, 2], [3, 4], [5, 6]])

# Print out its shape
# Same as x.size()

# Print out the number of elements in a particular dimension
# 0th dimension corresponds to the rows
x.shape[0] 

# Get the size of the 0th dimension
x.size(0)

# Example use of view()
# x_view shares the same memory as x, so changing one changes the other
x_view = x.view(3, 2)
print(x_view)

# We can ask PyTorch to infer the size of a dimension with -1
x_view = x.view(-1, 3)

# Change the shape of x to be 3x2
# x_reshaped could be a reference to or copy of x
x_reshaped = torch.reshape(x, (2, 3))

# Initialize a 5x2 tensor, with 5 rows and 2 columns
x = torch.arange(10).reshape(5, 2)