from libraries import *

data = [
    [0,1],
    [2,3],
    [4,5]

]

x_python = torch.tensor(data)

# print the tensor
print(x_python)

x_float = torch.tensor(data, dtype=torch.float)
print(x_float)

# We are using the dtype to create a tensor of particular type
x_bool = torch.tensor(data, dtype=torch.bool)