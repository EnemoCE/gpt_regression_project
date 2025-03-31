
import random
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

GENERATE_BATCHES = True
DATA_SIZE = 50000


def generate_regression_data(num_tasks, curriculum):
    data = []
    d = curriculum.n_dims_truncated
    fd = curriculum.n_dims_schedule.end
    t = curriculum.n_points
    for _ in range(num_tasks):
        w = torch.cat((torch.randn(d), torch.zeros(fd-d)))  # Sample w' from N(0, I)
        for _ in range(t):
            x = torch.cat((torch.randn(d), torch.zeros(fd-d)))  # Sample x_i from N(0, I)
            y = torch.dot(w, x)  # Calculate y_i = w' * x_i
            #y = torch.tensor(3.)
            data.append(x)
            data.append(torch.cat((y.unsqueeze(0), (torch.zeros(fd-1)))))  # y_i with 19 zeros and y_i'
    return torch.stack(data)  # Stack into a 2D tensor (T, C)


def generate_regression_data_fixed_w(num_tasks, curriculum):
    data = []
    d = curriculum.n_dims_truncated
    fd = curriculum.n_dims_schedule.end
    t = curriculum.n_points
    w = torch.cat((torch.randn(d), torch.zeros(fd-d)))  # Sample w' from N(0, I)
    for _ in range(num_tasks):
        for _ in range(t):
            x = torch.cat((torch.randn(d), torch.zeros(fd-d)))  # Sample x_i from N(0, I)
            y = torch.dot(w, x)  # Calculate y_i = w' * x_i
            #y = torch.tensor(3.)
            data.append(x)
            data.append(torch.cat((y.unsqueeze(0), (torch.zeros(fd-1)))))  # y_i with 19 zeros and y_i'
    return torch.stack(data)  # Stack into a 2D tensor (T, C)



import torch
import torch.nn as nn
import numpy as np
import random

# Determine the device to use (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

GENERATE_BATCHES = True
DATA_SIZE = 50000


def set_seed(seed):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA, set the seed for the CUDA random number generator as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    
    # Ensure deterministic behavior for certain operations on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def generate_regression_data(num_tasks, curriculum):
#     data = []
#     d = curriculum.n_dims_truncated
#     fd = curriculum.n_dims_schedule.end
#     t = curriculum.n_points

#     for task_idx in range(num_tasks):
#         # Define a small, randomly initialized neural network for the current task
#         seed = random.randint(1, 10**9)
#         set_seed(seed)
#         model = nn.Sequential(
#             nn.Linear(d, d // 2),
#             nn.Tanh(),             # Changed from ReLU to Tanh for symmetric output
#             nn.Linear(d // 2, 1)
#         ).to(device)
        
#         # Optionally, initialize weights to control output scale
#         def init_weights(m):
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#         model.apply(init_weights)

#         model.eval()

#         with torch.no_grad():  # Disable gradient computation for efficiency
#             for point_idx in range(t):
#                 # Generate input `x` with padding
#                 x_core = torch.randn(d, device=device)  # Move to device immediately
#                 x_padding = torch.zeros(fd - d, device=device)  # Move to device
#                 x = torch.cat((x_core, x_padding))  # Already on device

#                 # Prepare input for the neural network
#                 x_input = x_core.unsqueeze(0)  # Shape: [1, d]
#                 x_input = x_input.to(device)    # Ensure x_input is on the same device as the model

#                 # Compute output `y` using the neural network
#                 y = model(x_input).squeeze(0)  # Shape: [1] -> scalar

#                 # Pad `y` with zeros to match the desired dimension `fd`
#                 y_padded = torch.cat((y*3.2, torch.zeros(fd - 1, device=device)))  # Ensure on device

#                 # Append the input `x` and output `y_padded` to the data list
#                 data.append(x)
#                 data.append(y_padded)

#     # Stack all data points into a single 2D tensor
#     return torch.stack(data)  # Shape: (num_tasks * n_points * 2, fd)


# def generate_regression_data_fixed_w(num_tasks, curriculum):
#     data = []
#     d = curriculum.n_dims_truncated
#     fd = curriculum.n_dims_schedule.end
#     t = curriculum.n_points
#     # Define a small, randomly initialized neural network for the current task
#     seed = random.randint(1, 10**9)
#     set_seed(seed)
#     model = nn.Sequential(
#         nn.Linear(d, d // 2),
#         nn.Tanh(),             # Changed from ReLU to Tanh for symmetric output
#         nn.Linear(d // 2, 1)
#     ).to(device)
    
#     # Optionally, initialize weights to control output scale
#     def init_weights(m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
    
#     model.apply(init_weights)

#     model.eval()

#     for task_idx in range(num_tasks):

#         with torch.no_grad():  # Disable gradient computation for efficiency
#             for point_idx in range(t):
#                 # Generate input `x` with padding
#                 x_core = torch.randn(d, device=device)  # Move to device immediately
#                 x_padding = torch.zeros(fd - d, device=device)  # Move to device
#                 x = torch.cat((x_core, x_padding))  # Already on device

#                 # Prepare input for the neural network
#                 x_input = x_core.unsqueeze(0)  # Shape: [1, d]
#                 x_input = x_input.to(device)    # Ensure x_input is on the same device as the model

#                 # Compute output `y` using the neural network
#                 y = model(x_input).squeeze(0)  # Shape: [1] -> scalar

#                 # Pad `y` with zeros to match the desired dimension `fd`
#                 y_padded = torch.cat((y*3.2, torch.zeros(fd - 1, device=device)))  # Ensure on device

#                 # Append the input `x` and output `y_padded` to the data list
#                 data.append(x)
#                 data.append(y_padded)

#     # Stack all data points into a single 2D tensor
#     return torch.stack(data)  # Shape: (num_tasks * n_points * 2, fd)




def get_batch(curriculum, batch_size, iter=None):
    block_size = 2*curriculum.n_points
    if GENERATE_BATCHES:
      data = generate_regression_data(batch_size + 1, curriculum)
      ix = range(batch_size)
    else:
      data = generate_regression_data(DATA_SIZE, curriculum)
      ix = torch.randint(len(data)//block_size-1, (batch_size,))
    x = torch.stack([data[i*block_size:(i+1)*block_size] for i in ix])
    y = torch.stack([data[i*block_size+1:(i+1)*block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y






def get_fixed_weight_batch(curriculum, batch_size):
    block_size = 2*curriculum.n_points
    if GENERATE_BATCHES:
      data = generate_regression_data_fixed_w(batch_size + 1, curriculum)
      ix = range(batch_size)
    else:
      data = generate_regression_data_fixed_w(DATA_SIZE, curriculum)
      ix = torch.randint(len(data)//block_size-1, (batch_size,))
    x = torch.stack([data[i*block_size:(i+1)*block_size] for i in ix])
    y = torch.stack([data[i*block_size+1:(i+1)*block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
