# Oxygen diffusion in thermoelectric materials 

**oxidation** is a python implementation of a finite element model for the high-temperature oxidation of half-Heusler thermoelectric materials, with and without protective coatings.

Oxygen diffusion is solved with [FEniCSx](https://github.com/FEniCS) in a 1D coating/substrate domain, using a time-decaying diffusivity in each layer:

$$ \frac{\partial c}{\partial t} = \nabla\cdot\left( D \nabla c \right), \qquad D(t) = D_0 e^{-kt} $$

where $c$ is the normalised oxygen content. The mass gain is computed from the oxygen taken up by the material:

$$ m_f(t) = \rho_m \, p \int c \, \mathrm{d}x $$

The parameters $D_0$, $k$ and $p$ are calibrated against experimental mass-gain data and then used to extrapolate oxide-scale growth and the effect of coating thickness.

## Citation

This repository contains the numerical implementation of the results in the article published in *Modelling and Simulation in Materials Science and Engineering*. The article is available online from [this link](https://doi.org/10.1088/1361-651X/aeb95c).

```bibtex
@article{ekici2026oxidation,
  title={Modelling oxidation kinetics of half-Heusler thermoelectric materials and the role of protective coatings},
  author={Ekici, Ekrem and Gurtaran, Mikdat},
  journal={Modelling and Simulation in Materials Science and Engineering},
  year={2026},
  doi={10.1088/1361-651X/aeb95c},
  publisher={IOP Publishing}
}
```

## Installation

### Docker
The simplest option is to use the DOLFINx Docker image. Clone the repository and start a container in it:

```shell
git clone https://github.com/ekremekc/oxidation.git
cd oxidation
sudo docker run -ti -v $(pwd):/root/shared -w /root/shared --name=oxidation dolfinx/dolfinx:[vX.Y.Z]
```

The scripts also need the `diffusio2n` package, which can be installed in the container (in the repository directory) with:

```shell
pip3 install -e .
```

### Conda

This library can be installed directly with conda. We can make a conda environment for FEniCSx and then install this library into it.

```shell
conda create -n diphydro python=3.13.0 -y
conda activate diphydro
conda install pip
conda install -c conda-forge fenics-dolfinx=0.9.0 pyvista=0.44.1 -y # Linux and macOS
pip3 install -e .
```

Next, we run the following command in order to activate the environment:

```shell
conda activate diphydro
```

## How to use oxidation?

The cases in the paper are in the */paper* folder:

- */paper/onedim*: calibration (`main_N.py`, `main_P.py`, `main_N_coated.py`, `main_P_coated.py`), long-term extrapolation (`*_extrapolation.py`) and coating thickness (`*_coating_thickness.py`) runs
- */paper/verification*: comparison with the analytical solution
- */paper/dependency*: mesh and time-step convergence studies

Navigate to the case you like and type

```shell
python3 -u main_N.py
```

in the terminal. The results are written to *ResultsDir/fem_data*, and the plotting scripts in *ResultsDir* (for example `python3 plot_N.py`) generate the figures.
