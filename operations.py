import torch
from libraries import *

# Create an example tensor
x = torch.ones((3,2,2))

# Perform elementwise addition
# Use - for subtraction
x + 2

# Perform elementwise multiplication
# Use / for division
x * 2

# Create a 4x3 tensor of 6s
a = torch.ones((4,3)) * 6
a

# Create a 1D tensor of 2s
b = torch.ones(3) * 2
b

# Divide a by b
a / b

# Alternative to a.matmul(b)
# a @ b.T returns the same result since b is 1D tensor and the 2nd dimension
# is inferred
a @ b 

pp.pprint(a.shape)
pp.pprint(a.T.shape)

# Create an example tensor
m = torch.tensor(
    [
     [1., 1.],
     [2., 2.],
     [3., 3.],
     [4., 4.]
    ]
)

pp.pprint("Mean: {}".format(m.mean()))
pp.pprint("Mean in the 0th dimension: {}".format(m.mean(0)))
pp.pprint("Mean in the 1st dimension: {}".format(m.mean(1)))

# Create an example tensor
m = torch.tensor(
    [
     [1., 1.],
     [2., 2.],
     [3., 3.],
     [4., 4.]
    ]
)

pp.pprint("Mean: {}".format(m.mean()))
pp.pprint("Mean in the 0th dimension: {}".format(m.mean(0)))
pp.pprint("Mean in the 1st dimension: {}".format(m.mean(1)))

# Concatenate in dimension 0 and 1
a_cat0 = torch.cat([a, a, a], dim=0)
a_cat1 = torch.cat([a, a, a], dim=1)

print("Initial shape: {}".format(a.shape))
print("Shape after concatenation in dimension 0: {}".format(a_cat0.shape))
print("Shape after concatenation in dimension 1: {}".format(a_cat1.shape))

# add() is not in place
a.add(a)
a

# add_() is in place
a.add_(a)
a