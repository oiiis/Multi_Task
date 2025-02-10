import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

def adjust_learning_rate(optimizer, epoch):
    """Adjust learning rate based on your chosen schedule or performance metrics."""
    new_lr = initial_lr * (decay_rate ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
