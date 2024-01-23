# GSmoothSGD
Code for "Gaussian smoothing stochastic gradient descent"

The provided `requirements.yml` file can be used to create a conda environment to run this code.

## Config Info

- `env_name`: either 'mnist' or 'cifar10'
- `optimizer`: any combination of 'sgd', 'svrg', 'gssgd', 'gssvrg', 'gssvrg_tau', where `_tau` indicates that the control variate for GSmoothSVRG is the gradient of the original function ($\tau=0$ uses original gradient and $\tau=1$ uses smoothed gradient).
- `num_sgd`: number of SGD samples for gradient computation
- `inner_svrg_num_sgd`: number of SGD samples for the inner loop of SVRG (essentially `num_sgd` for SVRG)
- `num_mc_fsigma`: number of Monte Carlo points used in approximation of $\nabla f_{\sigma}$
- `inner_svrg_iterations`: number of iterations between control variate update for SVRG
- `s`: $\sigma$
