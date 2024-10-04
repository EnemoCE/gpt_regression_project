import torch
from task_sampler import get_batch
from models import IteratedNewtonModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'



@torch.no_grad()
def estimate_loss(model, args):
    curriculum = args.curriculum
    eval_iters = args.training.eval_iters
    batch_size = args.training.batch_size

    out = {}
    model.eval()
    losses = torch.zeros(eval_iters, 2)
    for k in range(eval_iters):
        X, Y = get_batch(curriculum, batch_size)
        logits1, loss1, = model(X, Y)
        logits2, loss2, = model(X, Y, base_model=False)
        losses[k] = torch.tensor([loss1.item(), loss2.item()])
    out = torch.mean(losses, 0)
    model.train()
    return out


@torch.no_grad()
def evaluate_test_task(model, curriculum, count, logs_ch=2):
    model.eval()
    x, targets = get_batch(curriculum, count)
    base_model = True if logs_ch == 1 else False
    logits, loss= model(x, targets, base_model=base_model)
    targets = targets[:, ::2, 0]
    B, TC = targets.shape
    logits = logits.view(B, TC)
    final_errors = torch.square(targets-logits).mean(dim=0)
    model.train()
    return final_errors.tolist()

@torch.no_grad()
def evaluate_iterative_newton_test_task(curriculum, num_iterations, count):

    input_dim = curriculum.n_dims_truncated
    newton_model = IteratedNewtonModel(input_dim=input_dim, num_iterations=num_iterations)
    newton_model.to(device)
    x_train, y_train = get_batch(curriculum, batch_size=count)

    A = x_train[:, 1::2, :]
    y = y_train[:, ::2, 0]

    w_k = newton_model(A, y)

    x25 = A[:, -1, :]
    y25_pred = torch.einsum('ij,ij->i', x25, w_k)
    y25_true = y[:, -1]
    error25 = torch.square(y25_true - y25_pred).mean(dim=0).item()
    return error25
