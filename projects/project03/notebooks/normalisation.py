import torch

norm_params = torch.load('./model_stats/normalisation_train_param.pt')
mu = norm_params['mu']
sigma = norm_params['sigma']
    
# Add methods
def preprocess(x):
    return (x - mu) / sigma

def postprocess(x):
    return sigma * x + mu

def print_xd(text):
    return text

