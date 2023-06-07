import numpy as np
from libraries import *

data = [
    [0,1],
    [2,3],
    [4,5]
]

# Initialize a tensor from a NumPy array
ndarray = np.array(data)
x_numpy = torch.from_numpy(ndarray)

# Print the tensor
print(x_numpy)