# lets' perform some autograd

from libraries import *

# Create an example tensor
# requires_grad parameter tells PyTorch to store gradients
x = torch.tensor([2.], requires_grad=True)

# Print the gradient if it is calculated
# Currently None since x is a scalar
pp.pprint(x.grad)

y = x * 3 # 3x^2
y.backward() # Calculate the gradient
pp.print(x.grad)