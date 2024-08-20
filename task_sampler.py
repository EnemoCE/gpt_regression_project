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



def get_batch(curriculum, batch_size):
    # generate a small batch of data of inputs x and targets y
    block_size = 2*curriculum.n_points
    if GENERATE_BATCHES:
      data = generate_regression_data(batch_size, curriculum)
      ix = range(batch_size-1)
    else:
      data = generate_regression_data(DATA_SIZE, curriculum)
      ix = torch.randint(len(data)//block_size-1, (batch_size,))
    x = torch.stack([data[i*block_size:(i+1)*block_size] for i in ix])
    y = torch.stack([data[i*block_size+1:(i+1)*block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y