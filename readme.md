# Simple Information-theoretic diffusion

A repository for the ICLR 2023 paper 
[here](https://openreview.net/forum?id=UvmDCdSPDOW)
containing simplified examples. 


## Requirements

To install requirements:

```setup
pip install numpy scipy matplotlib scikit-learn notebook torch pytorch-lightning
```
Pytorch lightning is used to simplify boilerplate, 
and makes it easier to exploit different compute resources. 


## Train and evaluate 2D examples

To train and generate figures, run this notebook. 

```train
jupyter notebook 2d_example.ipynb
```

The diffusion model takes a denoising architecture as input (arguments are input and log SNR, output is size of input).
This simplified code assumes continuous density estimation, and requires specifying the log SNR range 
(can be estimated via data spectrum as shown in our paper)

Pytorch lightning automatically checkpoints and logs with tensorboard. Don't forget to turn it on for visualization:  
```log
tensorboard --logdir .
```
Optionally, you can do "python train.py", 
and then load the checkpoint in the IPython notebook (optional cell included) to show the visualizations.

### Fixing the log SNR sampling parameters
This is the equivalent step to fixing the noise scheduler in other papers. 
Our analysis suggests that a two parameter log logistic distribution for 
log SNR is appropriate. We need to fix the mean and scale of this distribution. 
In the ICLR paper, we show how to use the spectrum of the data
to estimate appropriate values. For large datasets, this may be intractable.
Instead, we can treat these two values as hyper-parameters. 
However, rather than fix them blindly, we can estimate them empirically from one 
run of the diffusion model. 

To empirically set these hyper-parameters. Run the diffusion model once with the 
default values, and then use tensorboard to look at the MSE plot. 

TODO: picture of MSE curve
We want to pick the location, logsnr_loc to be the 50 percent quantile, 
and the scale to be the distance to the 75th percent quantile. Note the maximum value 
for MSE should be d. 


## Train on CIFAR-10

```train
python train_cifar10.py
```
MSE curves and log likelihoods are tracked in tensorboard. 
Checkpoints are saved automatically by Pytorch Lightning

## Results

Some of the following results appear in the notebook. 

### 2D Spiral

As an initial experiment, we consider the following 2D distribution:

![gauss2d](./Figs/samples.png)

### CIFAR-10 log-likelihood

We achieve an expected log likelihood of ... 
Note that the numbers in the notebook do not include ensembling, as in the paper. 
Also we estimate the continuous log-likelihood, while previous work focuses on discrete probability. 
