# closed-loop-coevolution
Simulations exploring a novel experimental approach to phage-bacteria coevolution, and the control systems required.  

## Usage
### Installation
```bash
# recommended (conda-based) reproducible install
conda create -n clc -c conda-forge python=3.12
conda activate clc
# install binary deps & common libs
conda install -c conda-forge ca-certificates openssl numpy pandas matplotlib seaborn ipykernel jax scipy
# install other deps & editable package via pip
python -m pip install jax-tqdm
python -m pip install -e .
# run an example simulation
python supplementary/example_simulation.py
```

### Running scripts
Once you have installed the package, you can generate the graphs using (for example):
```bash
python figures/figure_noevolution.py
```
This should take under a minute.

### MOI Simulations
The effect of Multiplicity of Infection (MOI) can be probed by changing the plant model referenced by `Plant.ode_model` in `plant_model.py`.