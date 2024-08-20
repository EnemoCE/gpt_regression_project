import torch
from task_sampler import get_batch


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
        logits, logits2, loss, loss2 = model(X, Y, eval=True)
        losses[k] = torch.tensor([loss.item(), loss2.item()])
    out = torch.mean(losses, 0)
    model.train()
    return out


@torch.no_grad()
def evaluate_test_task(model, curriculum, count, logs_ch=2):
    model.eval()
    x, targets = get_batch(curriculum, count)
    logits, logits2, loss, loss2 = model(x, targets, eval=True)
    if logs_ch == 2:
        logits = logits2
    targets = targets[:, ::2, 0]
    B, TC = targets.shape
    logits = logits.view(B, TC)
    final_errors = torch.square(targets-logits).mean(dim=0)
    model.train()
    return final_errors.tolist()

