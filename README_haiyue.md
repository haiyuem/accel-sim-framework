# Get accelsim dir:

```
git clone https://github.com/haiyuem/accel-sim-framework
```

## Get GPU apps (cutlass streamk):

```
git clone https://github.com/haiyuem/gpu-app-collection
# switch to micro26 branch 
git checkout micro26

#checkout submodule 
git submodule update --init --remote src/cuda/cutlass-bench

#make sure that it's on the right branch with the streamk changes 
cd src/cuda/cutlass-bench
git checkout micro26_e51efbfe
```

## Build streamk:

```
# need a della gpu node for this 
ssh della-gpu

#build in accelsim folder 
cd <accel-sim-framework>

#set env and build 
module load cudatoolkit/12.8 \
&& export CUDA_INSTALL_PATH=/usr/local/cuda-12.8 \
&& export PATH=$CUDA_INSTALL_PATH:$PATH \
&& source <gpu-app-collection>/src/setup_environment \
&& make -j -C <gpu-app-collection>/src cutlass_examples_ampere_gemm_streamk

## cutlass_examples_ampere_gemm_streamk defined in <gpu-app-collection>/src/Makefile
```

## Generate trace:

```
# Run the applications with the tracer (remember you need a real GPU for this):
ssh della-gpu
cd <accel-sim-framework>

./util/tracer_nvbit/run_hw_trace.py -B gemm_streamk -D 0 
# the apps are defined in util/job_launching/apps/define-all-apps.yml
# trace will be generated in ./hw_run/traces/
```

## Setup apptainer to run workloads on della:

```
#taken from https://github.com/accel-sim/Dockerfile
singularity pull docker://ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8
# should generate a .sif file 
```

## Run scripts for AccelSim:

```
# all my scripts are in this repo, feel free to only take the scripts relevant for you 
git clone https://github.com/haiyuem/differential_architecture

# set up your env paths in run_cmds/AccelSim/global_env.sh

# setup slurm job run scripts for each job in sweep - not actually launching jobs 
# you need cuda env for this, but can run on login node  
ssh della-gpu
./run_cmds/AccelSim/SM80_A100/launch_accelsim.sh

# actually launch jobs on della slurm cpu jobs - without cuda env 
# to work around this, I created a new env file that doesn't require cuda env 
# feel free to sync my gpgpusim folder https://github.com/haiyuem/gpgpu-sim_distribution for accel-sim-framework/gpu-simulator/gpgpu-sim (switch to branch micro26); my only change so far is https://github.com/haiyuem/gpgpu-sim_distribution/blob/micro26/setup_environment_no_cuda, which I use in accel-sim-framework/util/job_launching/slurm.sim

./run_cmds/AccelSim/SM80_A100/relaunch_incomplete.sh

# this script checks each result dir and launches a new slurm job if no result exists. Can also be used to relaunch failed jobs. 
```

## Plotting scripts for results: 

```
# very preliminary AI-generated plotting scripts
run_cmds/AccelSim/collect_results.ipynb
```

