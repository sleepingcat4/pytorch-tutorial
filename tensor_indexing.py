import torch

# Initialize an example tensor
x = torch.Tensor([
                  [[1, 2], [3, 4]],
                  [[5, 6], [7, 8]], 
                  [[9, 10], [11, 12]] 
                 ])

# Access the 0th element, which is the first row
x[0] # Equivalent to x[0, :]

# Get the top left element of each element in our tensor
x[:, 0, 0]

# Let's access the 0th elements of the 1st and 2nd elements
i = torch.tensor([1, 2])
j = torch.tensor([0])
x[i, j]