# BayDON-HMC

**Bayesian Parameter Identification for PDEs using Deep Operator Networks and Hamiltonian Monte Carlo**

This script identifies an unknown physical parameter in a partial differential equation (PDE)
from sparse noisy sensor measurements, and quantifies the full uncertainty in that estimate
using Bayesian inference. No knowledge of the true parameter is required at inference time —
only the sensor readings.

---

## What does this do?

Imagine you have a physical system governed by the 1D diffusion (heat) equation:

```
∂u/∂t = ξ · ∂²u/∂x²
```

where `ξ` is an unknown diffusion coefficient — something like thermal conductivity or
molecular diffusivity that you cannot measure directly. You only have access to noisy
readings from a small number of sensors placed at fixed locations in the system.

This script learns to:
1. Reconstruct the full solution field from sparse sensor readings (using a Deep Operator Network)
2. Estimate a probability distribution over `ξ` from those same sensor readings (using a variational neural network)
3. Refine that distribution into a full Bayesian posterior using Hamiltonian Monte Carlo sampling

The output is not just a single guess for `ξ` — it is a full probability distribution
telling you both the most likely value and how uncertain that estimate is.

---

## Who is this for?

This script is intended for researchers and students in scientific machine learning,
computational physics, or engineering who want to:

- Identify unknown parameters in PDEs from sparse data
- Quantify uncertainty in those estimates
- Understand how deep learning and Bayesian inference can be combined for inverse problems

Basic familiarity with Python is assumed. You do not need prior experience with
deep learning or Bayesian statistics to run the script, but the comments throughout
explain the key ideas.

---

## Requirements

### Python version

Python 3.9 or higher is recommended. You can check your version by running:

```bash
python --version
```

### Dependencies

The script requires three libraries:

```
torch       — deep learning framework (neural networks, automatic differentiation)
numpy       — numerical arrays and math
matplotlib  — plotting
```

Install them all at once with:

```bash
pip install torch numpy matplotlib
```

If you are new to Python and do not have `pip`, see
[https://pip.pypa.io/en/stable/installation](https://pip.pypa.io/en/stable/installation).

### GPU (optional)

The script automatically uses a GPU if one is available. If not, it runs on CPU.
No changes to the code are needed — this line handles it automatically:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Running on CPU is fine for this problem but will be slower, particularly during
the derivative precomputation step.

---

## Installation

No installation is required beyond the dependencies above.

Download or clone the repository:

```bash
git clone https://github.com/<your-username>/baydon-hmc.git
cd baydon-hmc
```

Or simply download `baydon_hmc.py` directly and place it in a folder of your choice.

---

## How to run

```bash
python baydon_hmc.py
```

That is it. The script generates all data synthetically, trains the networks,
runs inference, and saves all outputs automatically. You do not need to prepare
any dataset.

### What happens when you run it

The script proceeds through four stages, printing progress to the terminal:

**Stage 1 — Generate data**
Synthetic training data is created by sampling 800 random values of `ξ` from the
range `[0.05, 0.5]` and computing the analytic solution at sensor locations and
on a dense query grid. Gaussian noise is added to simulate realistic sensor readings.

**Stage 2 — Train DeepONet (Gθ)**
A Deep Operator Network is trained to learn the mapping from sparse sensor readings
to the full solution field. This takes the most time. Progress is printed every
200 epochs. Expected output looks like:

```
Epoch  200 | train MSE 0.003560 | val MSE 0.003808
Epoch  400 | train MSE 0.001893 | val MSE 0.001932
...
Epoch 6000 | train MSE 0.000174 | val MSE 0.000186
```

The model with the best validation MSE is automatically saved and reloaded.

**Stage 3 — Train Identifier MLP (Iφ)**
A smaller network is trained to map sensor readings directly to a Gaussian
probability distribution over `ξ`. Progress is printed every 300 epochs:

```
Epoch  300 | -ELBO 15.3566 | val MAE 0.0914
Epoch  600 | -ELBO  5.7110 | val MAE 0.0497
...
```

The `-ELBO` (negative Evidence Lower Bound) should decrease over training.
The `val MAE` measures how accurately the network predicts `ξ` on held-out data.

**Stage 4 — HMC inference**
Given a new test observation (a noisy sensor reading generated from the true
`ξ* = 0.18`), the script evaluates the Identifier network to get a prior
distribution, then runs Hamiltonian Monte Carlo to produce a full posterior
distribution. The final summary looks like:

```
HMC acceptance rate   = 89.06%
Posterior mean        = 0.1877  (true: 0.1800)
Posterior std         = 0.0252
95% credible interval = [0.1392, 0.2370]
True xi* in CI?       = True
```

---

## Outputs

After running, the following files are created:

```
figures/
    01_gtheta_prediction_t005.png   DeepONet prediction vs truth at t=0.05
    01_gtheta_prediction_t02.png    DeepONet prediction vs truth at t=0.20
    01_gtheta_prediction_t04.png    DeepONet prediction vs truth at t=0.40
    02_gtheta_training_loss.png     DeepONet training and validation MSE curves
    03_iphi_mu_vs_xi.png            Identifier predicted mean vs true ξ (validation)
    04_iphi_training.png            Identifier ELBO and MAE training curves
    05_posterior_histogram.png      HMC posterior density with 95% credible interval
    06_hmc_trace.png                HMC chain trace plot
```

The `figures/` folder is created automatically if it does not exist.

---

## Understanding the outputs

### DeepONet prediction plots
These show how well the neural network reconstructs the solution field from
just 10 noisy sensor readings at each time slice. The black curve is the true
solution, the dashed teal curve is the network prediction, and the red dots
are the noisy sensor readings used as input.

### Identifier training plot
The left axis shows the ELBO loss decreasing over training — lower is better.
The right axis shows the validation MAE, which measures how accurately the
network predicts `ξ` from sensor patterns — lower is better.

### Posterior histogram
This is the main result. It shows the full probability distribution over `ξ`
produced by HMC. The shaded region is the 95% credible interval — the range
that contains the true `ξ` with 95% probability under the model. If this
interval contains the true value (marked by the black vertical line), the
inference is working correctly.

### HMC trace plot
This shows the sequence of `ξ` values sampled by HMC after burn-in. A healthy
trace should look like random noise around a stable mean — no long drifts or
getting stuck in one place. This confirms the sampler is mixing well.

---

## Key settings

All tunable parameters are at the top of `baydon_hmc.py` under the
`GLOBAL CONFIGURATION` section. The most important ones are:

| Parameter | Default | What it controls |
|---|---|---|
| `XI_MIN / XI_MAX` | 0.05 / 0.5 | Range of ξ values in training data |
| `N_TRAIN_SAMPLES` | 800 | Number of training examples |
| `OBS_NOISE_STD` | 0.05 | Standard deviation of sensor noise |
| `EPOCHS_DEEPONET` | 6000 | How long to train the DeepONet |
| `EPOCHS_IDENT` | 5000 | How long to train the Identifier |
| `HMC_N_SAMPLES` | 3000 | Number of posterior samples from HMC |
| `XI_STAR` | 0.18 | True ξ value used for the test observation |

To test a different true value, change `XI_STAR` in the inference section.
To speed up a test run at the cost of accuracy, reduce `EPOCHS_DEEPONET`
to 2000 and `N_TRAIN_SAMPLES` to 200.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
Run `pip install torch` and try again.

**The training loss is not decreasing / flat from the start**
This usually means the learning rate is too high or the network is not
initialising well. Try rerunning — random initialisation sometimes produces
a bad starting point. If the problem persists, reduce `LR_DEEPONET` from
`5e-4` to `1e-4`.

**The posterior is very wide and the CI spans almost the entire ξ range**
This means the DeepONet did not train well enough — its derivatives are
too noisy to give the Identifier a useful physics signal. Increase
`EPOCHS_DEEPONET` or check that the training loss reached below `1e-3`.

**The script runs but no figures appear**
The script saves figures to `figures/` rather than displaying them interactively.
Check the `figures/` folder in the same directory where you ran the script.

**Out of memory error**
Reduce `BATCH_SIZE` from 64 to 32 or 16.

---

