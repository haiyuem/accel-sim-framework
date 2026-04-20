# AccelSim command cheatsheet

Personal reference for Haiyue's AccelSim workflow on Della. Fork lives at <https://github.com/haiyuem/accel-sim-framework>. For the broader project layout, see `../../../CLAUDE.md`; for NCU profiling, see `ncu/README.md`.

All paths below assume `$ACCELSIM = /scratch/gpfs/WENTZLAF/hm2595/differential_arch/simulators/AccelSim/accel-sim-framework` and `$APPS = $ACCELSIM/../gpu-app-collection`.

## 1. Environment (Della)

```bash
module load cudatoolkit/12.8
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
```

Use `della-gpu` (or any `della-w2` GPU node) whenever a real GPU is required — that's building CUDA apps, running the tracer, and profiling with NCU. `della-cpu` is fine for the Accel-Sim binary build and for simulation runs (they execute inside the Apptainer image without CUDA).

The Apptainer image is at `$ACCELSIM/../accel-sim-framework_ubuntu-24.04-cuda-12.8.sif`. For interactive debugging:

```bash
apptainer exec $ACCELSIM/../accel-sim-framework_ubuntu-24.04-cuda-12.8.sif /bin/bash
```

## 2. Build the simulator

```bash
cd $ACCELSIM
source ./gpu-simulator/setup_environment.sh

# CMake build (preferred)
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build
cmake --build ./gpu-simulator/build -j8
cmake --install ./gpu-simulator/build

# or Make
make -j -C ./gpu-simulator
```

Binary: `./gpu-simulator/bin/release/accel-sim.out`.

## 3. Build a benchmark from `gpu-app-collection`

Needs a GPU node (`ssh della-gpu`) because nvcc compiles against the real device.

```bash
cd $ACCELSIM
source $APPS/src/setup_environment
make -j -C $APPS/src <target>
```

Built binaries land in `$APPS/bin/12.8/release/`. Targets used in this project:

| Target | Binary | Notes |
| --- | --- | --- |
| `rodinia_2.0-ft` | many | functional-test set used by upstream short tests |
| `ubench_l2_bw_128` | `l2_bw_128` | single-knob L2 bandwidth microbench |
| `cutlass_examples_ampere_gemm_streamk` | `gemm_streamk` | CUTLASS StreamK GEMM; source at `$APPS/src/cuda/cutlass-bench/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu`; default config runs `run<DeviceGemmStreamK>("StreamK GEMM with default load-balancing", options)` |
| `shoc-FFT` | `shoc-FFT` | see §6 for the Makefile/yaml patch that wires it up |

`make -C $APPS/src data` pulls the Purdue data bundle — it's fetched once and referenced by apps that list `data_dirs`. Skip if you only care about apps that take no data files (e.g. `gemm_streamk`, `l2_bw_128`).

## 4. Trace a benchmark

Also needs a GPU node. The tracer is NVBit-based and must see a real device.

```bash
ssh della-gpu
cd $ACCELSIM
./util/tracer_nvbit/run_hw_trace.py -B <app-suite-in-define-all-apps.yml> -D 0
# traces land in ./hw_run/traces/device-0/12.8/<binary>/<args>/
```

Useful flags:

- `--dynamic_kernel_range "<range>"` — forwards a raw `DYNAMIC_KERNEL_RANGE` string, e.g. `4`, `4-4`, or `5-8@.*attention.*`. Overrides `--limit_kernel_number`.
- `--spinlock_handling fast_forward` if the app spins.

**Gotcha (compute-node PATH):** `nvdisasm` and `cuobjdump` aren't on Della's GPU compute nodes by default. They were copied once from a head node into `util/tracer_nvbit/cuda_tools/`, and `run_hw_trace.py` now prepends that dir to `PATH` in every generated trace script. If a new CUDA toolkit is installed, refresh those two binaries.

Examples:

```bash
./util/tracer_nvbit/run_hw_trace.py -B rodinia_2.0-ft -D 0
./util/tracer_nvbit/run_hw_trace.py -B l2_bw_128 -D 0
./util/tracer_nvbit/run_hw_trace.py -B gemm_streamk -D 0
./util/tracer_nvbit/run_hw_trace.py -B fused_mha -D 0 --dynamic_kernel_range 4
```

## 5. Run the simulator

From any node (CPU fine) once the trace exists:

```bash
cd $ACCELSIM
./util/job_launching/run_simulations.py \
  -C A100-SASS \
  -B <app-suite> \
  -T ./hw_run/traces/device-0/12.8 \
  -N <run-name>
```

- `-C` — config name in `util/job_launching/configs/define-standard-cfgs.yml`. Hyphen-joined names are composable (e.g. `A100-SASS-l2clock_882`). The first token pulls the base config from that YAML; for `A100` that resolves to `./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config`.
- `-B` — benchmark suite, defined in `util/job_launching/apps/define-all-apps.yml`.
- `-T` — traces root (the parent of the per-app trace dirs).
- `-N` — launch name, used as the directory tag and for `monitor_func_test.py` / `get_stats.py` lookups.

Per-job results land under `sim_run_12.8/<app>/<args>/<config>/<name>/`.

Monitor and collect:

```bash
./util/job_launching/monitor_func_test.py -v -N <run-name>
./util/job_launching/get_stats.py -N <run-name> | tee results.csv
```

## 6. Adding a new benchmark — shoc-FFT walkthrough

Edit `$APPS/src/Makefile` to add the target:

```makefile
shoc-FFT:
	mkdir -p $(BINDIR)/$(BINSUBDIR)/
	cd cuda/shoc-master/; ./configure; $(SETENV) make $(MAKE_ARGS);
	$(SETENV) make $(MAKE_ARGS) -C cuda/shoc-master/src/cuda/level1/fft
	mv cuda/shoc-master/src/cuda/level1/fft/FFT $(BINDIR)/$(BINSUBDIR)/shoc-FFT
```

Register the app in `util/job_launching/apps/define-all-apps.yml`:

```yaml
shoc-master:
    exec_dir: "$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/shoc-master/"
    data_dirs: "$GPUAPPS_ROOT/data_dirs/shoc-master/"
    execs:
        - fft:
            - args:
              accel-sim-mem: 2G
```

Then build → trace → run:

```bash
make -j -C $APPS/src shoc-FFT                                            # on della-gpu
./util/tracer_nvbit/run_hw_trace.py -B shoc-master -D 0                  # on della-gpu
./util/job_launching/run_simulations.py -C A100-SASS -B shoc-FFT \
  -T ./hw_run/traces/device-0/12.8 -N myTest3                            # any node
```

The `-B` name matches the YAML key.

## 7. Differential-arch sweeps

Don't drive `run_simulations.py` by hand for parameter sweeps — use the launcher at `run_cmds/AccelSim/SM80_A100/run_cmd.sh` (parent project). It edits `env.sh`, generates slurm scripts via `launch_accelsim.sh`, and backfills incomplete runs via `relaunch_incomplete.sh`. See the parent `CLAUDE.md` for the sweep schema and dynamic-reload knobs.
