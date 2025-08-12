import re
import torch
import tqdm
from torch.func import functional_call, vmap, grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def extract_hidden_substring(s):
    # define the regex pattern to match "hidden.X" where X is a number
    pattern = r'conv_block3\.\d+'
    
    # search for the pattern in the given string
    match = re.search(pattern, s)
    
    # if a match is found, return the matched substring
    if match:
        return match.group(0)
    else:
        return None

class LinearNet(torch.nn.Module):
	def __init__(self, n_hidden_layers=1, n_inputs=1, n_units=64, activation_fn=torch.nn.Tanh()):
		super(LinearNet, self).__init__()

		layers = []
		
		# input layer
		layers += [torch.nn.Linear(in_features=n_inputs, out_features=n_units), activation_fn]
		
		# adding hidden layers with activations (could this be replaced by listing modules?)
		# the final linear layer has no activation function (improves uncertainty estimates)
		layers += [torch.nn.Linear(in_features=n_units, out_features=n_units) if i %
				   2 == 0 else activation_fn for i in range((n_hidden_layers * 2) - 1)]
		
		# output layer is NIG
		layers += [torch.nn.Linear(in_features=n_units, out_features=1)]

		self.hidden = torch.nn.Sequential(*layers)

	def forward(self, x):
		x = self.hidden(x)
		return x
	
# # get tri-diagonal scale matrix / Compute ``M^{-0.5}`` as a tridiagonal matrix.
def _precision_to_scale_tril(P):
	# Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
	Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
	L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
	Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
	L = torch.linalg.solve_triangular(L_inv, Id, upper=False)

	return L

def _precision_to_scale_tril_robust(P, epsilon=1e-6):
    """
    Compute scale matrix (P^(-1/2)) using SVD decomposition.
    More robust version that handles non-PD matrices.
    
    Args:
        P: Precision matrix
        epsilon: Threshold for eigenvalue truncation/regularization
    
    Returns:
        L: Lower triangular scale matrix such that LL^T ≈ P^(-1)
    """
    # Compute SVD decomposition
    U, s, Vh = torch.linalg.svd(P, full_matrices=False)
    
    # Handle small/negative eigenvalues
    s_sqrt_inv = torch.where(s > epsilon, 
                            torch.sqrt(1.0 / s),
                            torch.zeros_like(s))
    
    # Compute P^(-1/2) = U * D^(-1/2) * V^H
    scale_matrix = U @ torch.diag(s_sqrt_inv) @ Vh
    
    # Convert to lower triangular form using Cholesky
    # Add small diagonal term for numerical stability
    scale_cov = scale_matrix @ scale_matrix.T
    scale_cov = (scale_cov + scale_cov.T) / 2  # Ensure symmetry
    scale_cov += epsilon * torch.eye(scale_cov.shape[-1], 
                                   dtype=scale_cov.dtype, 
                                   device=scale_cov.device)
    
    try:
        L = torch.linalg.cholesky(scale_cov)
    except RuntimeError:
        # If Cholesky fails, fall back to eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(scale_cov)
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=epsilon))
        L = eigvecs @ torch.diag(sqrt_eigvals)
    
    return L

def jacobian(model, x, filter, model_output_size=1):
	model.zero_grad()

	if len(filter) != 0:
		params = {k: v.detach() for k, v in model.named_parameters() if any([fltr in k for fltr in filter])}
		buffers = {k: v.detach() for k, v in model.named_buffers() if any([fltr in k for fltr in filter])}
	else:
		params = {k: v.detach() for k, v in model.named_parameters()}
		buffers = {k: v.detach() for k, v in model.named_buffers()}

	def compute_output(params, buffers, sample):		
		batch = sample.unsqueeze(0)

		# import ipdb; ipdb.set_trace()
		predictions = functional_call(model, (params, buffers), (batch,))
		return predictions.sum()
		# return predictions

	ft_compute_grad = grad(compute_output)
	ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0), randomness="same")
	# print(compute_output(params, buffers, x))
	if len(x.shape) < 4:
		x = x.unsqueeze(0)
		# import ipdb; ipdb.set_trace()
	ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x)

	# for k, v in model.named_parameters():
	# 	print(k, extract_hidden_substring(k))
	
	Jk = torch.concat([v.flatten(start_dim=1) for v in ft_per_sample_grads.values()], dim=1)
	# Necessary to detach so cude memory is not overloaded on iterations
	Jk = Jk.unsqueeze(-1).transpose(1, 2).detach()

	f = model(x)
	return Jk, f.detach()

def get_hessian(model, filter, train_loader, device):
	if len(filter) != 0:
		n_params = sum(v.numel() for k, v in model.named_parameters() if any([fltr in k for fltr in filter]))
	else:
		n_params = sum(v.numel() for _, v in model.named_parameters())
	H = torch.zeros(n_params, n_params).to(device)

	for X, _ in tqdm.tqdm(train_loader):
		Jk, _ = jacobian(model, X.to(device), filter=filter)
		H += torch.einsum('mkp,mkq->pq', Jk, Jk)

	prior_lambda = len(train_loader.dataset) / 64.
	H += prior_lambda * torch.eye(n_params).to(device)
	
	return H

def sample_weights(mu, sigma):
	return mu + sigma @ torch.randn_like(mu)

def glm_predictive_samples(model, x, posterior_covariance, filter, n_samples=100):
	# predict the output of the model
	Js, f_mu = jacobian(model, x, filter=filter)
	# Here is where it gets non-trivial, f_var is flattened, but predictions are of course not.
	# Can we simply reshape the outputs after sampling?
	f_var = torch.einsum('ncp,pq,nkq->nck', Js, posterior_covariance, Js)

	# import ipdb; ipdb.set_trace()
	# sample from the normal distribution
	return torch.distributions.MultivariateNormal(loc=f_mu.detach().view(16, -1), covariance_matrix=f_var.detach()).sample((n_samples,))

def _nn_predictive_samples(model, X, posterior_covariance, filter, n_samples=100):
    if len(filter) != 0:
        param_items = [(k, v) for k, v in model.named_parameters() if any([fltr in k for fltr in filter])]
    else:
        param_items = list(model.named_parameters())

    param_tensors = [v for _, v in param_items]
    mean = parameters_to_vector(param_tensors)
    device = next(model.parameters()).device

    fs = list()
    for _ in range(n_samples):
        sample = sample_weights(mean, posterior_covariance)
        vector_to_parameters(sample, param_tensors)
        predictions = model(X.to(device))
        fs.append(predictions.detach())

    vector_to_parameters(mean, param_tensors)
    fs = torch.stack(fs)

    return fs


def sample_weights(mu, sigma):
	return mu + sigma @ torch.randn_like(mu)
