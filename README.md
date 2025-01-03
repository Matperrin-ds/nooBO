<div style="text-align: center">
<img src="nooBO.webp" width="500">
</div>

# nooBO : Beginner-friendly Bayesian Optimization
> Run BoTorch-based Bayesian Optimization with (almost) a single line of code

## Nota Bene
nooBO is made for beginners who want to solve toy problems such as school assignments while understanding what happens under the hood. It wraps-up the few lines of code required to perform basic Bayesian Optimization using [`BoTorch`](https://botorch.org/), while automatically storing important metrics with [`Tensorboard`](https://www.tensorflow.org/tensorboard?hl=en) for easy vizualisation. With a couple of lines of code, it enables solving continuous optimization problems in low-dimensional spaces (roughly under 50 dimensions). For full-fledged research/engineering, it is best to use [`BoTorch`](https://botorch.org/), either directly or through [`Ax`](https://botorch.org/docs/botorch_and_ax) which offers a production-ready, higher level API for adaptive experimentation.  

## Installation
First install required dependencies.
```bash
pip install torch botorch tensorboard
```
Then clone the repository.
```
git clone https://github.com/Matperrin-ds/nooBO.git
```
## Usage
Import nooBO and some toy function, here we use Ackley which has a 2D input.
```python
from nooBO import *
from botorch.test_functions import Ackley
```
We will negate Ackley because it is supposed to be minimized but BoTorch natively seeks the argmax and not the argmin.
```python
objective_function = Ackley(dim=2, negate = True)
```
Then we specify the number of input dimensions, the optimization bounds and the number of initial random points to create the optimization problem.
```python
problem = optimization_problem(objective_function, num_dims = 2,
bounds = torch.Tensor([[-32.768, -32.768],[32.768, 32.768]]), n_randinit_points = 10)
```
Finally, we can run the Bayesian Optimization algorithm for 300 iterations.
```python
result = problem.run(n_iter=300)
```
Once the algorithm is finished running, we can visualize various metrics in tensorboard by firt running the following command
```bash
tensorboard --logdir runs
```
then opening a web browser and going to http://localhost:6006/.

## Technical details
nooBO performs Bayesian Optimization with [`Gaussian Processes`](https://gaussianprocess.org/gpml/chapters/RW.pdf) as the surrogate model. More specifically, it uses the [`Matérn 5/2`](https://docs.gpytorch.ai/en/v1.13/kernels.html) kernel and a learnable [`constant mean`](https://docs.gpytorch.ai/en/v1.13/means.html) (These are pretty common choices). At each iteration, the algorithm makes an educated guess for where to evaluate the objective fuction in a "smart" way by optimizing the [`Log Expected Improvement`](https://arxiv.org/abs/2310.20708) acquisition function via [`L-BFGS`](https://aria42.com/blog/2014/12/understanding-lbfgs).
