import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from botorch.models import SingleTaskGP, SingleTaskVariationalGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel, MaternKernel

from typing import Optional
from typing import Callable

from tqdm import tqdm


class optimization_problem():
    def __init__(self, objective_function: Callable, num_dims: int, bounds: Tensor, n_randinit_points: Optional[int] = 10,
                X_0: Optional[Tensor] = None, X_1: Optional[Tensor] = None, Y_1: Optional[Tensor]= None):
        """
        Initialize the optimization_problem class.

        Args:
            objective_function (Callable): The objective function to be optimized.
            num_dims (int): The number of dimensions of the input space.
            bounds (Tensor): The bounds of the input space.
            n_randinit_points (Optional[int], optional): The number of random initialization points. Defaults to 10.
            X_0 (Optional[Tensor], optional): Manually chosen locations where to evaluate the objective function. Defaults to None.
            X_1 (Optional[Tensor], optional): Manually chosen locations to include in the training set. Defaults to None.
            Y_1 (Optional[Tensor], optional): Function values corresponding to X_1. Defaults to None.
        """
        self.objective_function = objective_function
        self.train_X = torch.empty(0, num_dims, dtype=torch.float64)
        self.train_Y = torch.empty(0, dtype=torch.float64)
        self.writer = SummaryWriter()
        self.bounds = bounds

        # Random initialization points
        if n_randinit_points is not None:
            unit_random_X = torch.rand(n_randinit_points, num_dims, dtype=torch.float64)
            lower_bounds = self.bounds[0]
            upper_bounds = self.bounds[1]
            random_X = unit_random_X * (upper_bounds - lower_bounds) + lower_bounds
            for x in random_X:
                self.train_Y = torch.cat([self.train_Y, objective_function(x.unsqueeze(0))])
            self.train_X = torch.cat([self.train_X, random_X])
            
        # Manually chosen locations
        if X_0 is not None:
            for x in X_0:
                self.train_Y = torch.cat([self.train_Y, objective_function(x.unsqueeze(0))])
            self.train_X = torch.cat([self.train_X, X_0])

        # Manually chosen locations and corresponding function values
        if X_1 is not None and Y_1 is not None :
            if len(X_1) == len(Y_1):
                self.train_X = torch.cat([self.train_X, X_1])
                self.train_Y = torch.cat([self.train_Y, Y_1])
            else:
                raise ValueError(f"X_1 and Y_1 must have the same length. Received X_1 with length {len(X_1)} and Y_1 with length {len(Y_1)}.")
        
        # The GP requires at least 1 initial point
        if len(self.train_X) == 0:
            raise ValueError(f"At least one of n_randinit_points, X_0, X_1, and Y_1 must be provided. Received n_randinit_points = {n_randinit_points}, X_0 = {X_0}, X_1 = {X_1}, Y_1 = {Y_1}.")
        
        # Log some metrics in tensorboard for the initial points
        for i in range(1, len(self.train_X)+1):
            # Log objective values in tensorboard
            self.writer.add_scalar("Objective", self.train_Y[i-1], i)

            # Compute distance to previous candidates and log it in tensorboard
            if i >= 2:
                distances = torch.sqrt(torch.sum(((self.train_X)[0:i-1].detach() - (self.train_X)[i-1])**2, dim=1))
                min_distance = torch.min(distances)
                self.writer.add_scalar("Distance to previous candidates", min_distance.item(), i)

            # Log maximum so far
            self.writer.add_scalar("Convergence plot", torch.max((self.train_Y)[0:i]), i)
        

    def run(self, n_iter: int = 100, num_restarts: int = 5, raw_samples: int = 10000) -> dict:
        """
        Run the Bayesian optimization algorithm.

        Args:
            n_iter (int, optional): The number of iterations. Defaults to 100.
            gp_type (str, optional): The type of GP model to use. Defaults to 'full_rank'.
            n_inducing_points (Optional[int], optional): The number of inducing points for sparse GP. Defaults to 200.
            num_restarts (int, optional): The number of restarts for optimizing the acquisition function. Defaults to 5.
            raw_samples (int, optional): The number of Monte Carlo samples for optimizing the acquisition function. Defaults to 10000.

        Returns:
            dict: A dictionary containing the results of the optimization process.
        """
        pbar = tqdm(range(1+len(self.train_X), n_iter+1))
        for i in pbar:
            pbar.set_description(f"Performing Bayesian optimization: iteration {i}/{n_iter} - current best: {self.train_Y.max().item():.4f}")

            # Instantiate GP
            gp = SingleTaskGP(normalize(self.train_X, self.bounds), (self.train_Y).unsqueeze(1), outcome_transform=Standardize(m=1), covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.train_X.size(-1))))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            # This is for accessing the 'inner' kernel's hyperparameters (lengthscales) since the kernel is the composition of a scale kernel ('outer' part) and a Matérn kernel ('inner' part)
            underlying_gp = gp

            # Fit GP hyperparameters
            fit_gpytorch_mll(mll)

            # Log hyperparameters in tensorboard
            self.writer.add_scalar("Mean", underlying_gp.mean_module.constant.item(), i)
            self.writer.add_scalar("Outputscale", underlying_gp.covar_module.outputscale.item(), i)
            lengthscale = underlying_gp.covar_module.base_kernel.lengthscale.squeeze()
            for k in range(len(lengthscale)):
                self.writer.add_scalar(f"Lengthscale_{k}", lengthscale[k].item(), i)

            # Optimize acquisition function using L-BFGS-B with restarts
            acquisition_function = LogExpectedImprovement(gp, best_f=self.train_Y.max().item())
            candidate, acq_value = optimize_acqf(acquisition_function, bounds=self.bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples)
            observation = self.objective_function(candidate)

            # Log acquisition and objective values in tensorboard
            self.writer.add_scalar("Objective", observation, i)
            self.writer.add_scalar("Acquisition", acq_value, i)

            # Compute distance to previous candidates and log it in tensorboard
            distances = torch.sqrt(torch.sum((self.train_X.detach() - candidate)**2, dim=1))
            min_distance = torch.min(distances)
            self.writer.add_scalar("Distance to previous candidates", min_distance.item(), i)

            # Update training data
            self.train_X = torch.cat([self.train_X, candidate])
            self.train_Y = torch.cat([self.train_Y, observation])

            # Log maximum so far
            self.writer.add_scalar("Convergence plot", torch.max(self.train_Y), i)

        # Fetch max and argmax of objective function
        max_idx = torch.argmax(self.train_Y)
        max_X = self.train_X[max_idx]
        max_Y = self.train_Y[max_idx]

        # Log evaluation locations in tensorboard
        self.writer.add_embedding(self.train_X, metadata=[f'x_{i}' for i in range(self.train_X.shape[0])])
        self.writer.flush()
        self.writer.close()
        
        return {'X': self.train_X.detach().numpy(), 'Y': self.train_Y.detach().numpy(), 'argmax': max_X.detach().numpy(), 'max': max_Y.item()}


