# gpu_timed_run — timed GPU run of the 2-D BFS inverse problem

Runs the original Adam optimisation of [`2D-BFS/Diff_Physics.py`](../../2D-BFS/Diff_Physics.py)
on the GPU and times every stage of every epoch, with pre-flight checks and an ETA. It uses the
`gpu_*.py` modules of `2D-BFS-GPU/`; requirements and data as in [`../README.md`](../README.md).

```bash
cd 2D-BFS-GPU/gpu_timed_run
python run_gpu_timed.py                     # 10 epochs, N = 4000
python run_gpu_timed.py --epochs 200        # full run
python run_gpu_timed.py --N 200 --epochs 3  # smoke test
```

Options: `--epochs`, `--N`, `--data`, `--out`, `--cpu-epoch-s` (see the script header).
Outputs in `results_<E>ep_N<N>/`: `history.json`, `summary.json`, `timing.png`,
`inlet_control.h5`.

Time on an RTX A2000 (N = 4000): 5.7 s per epoch (Navier–Stokes 0.6 s, advection–diffusion
2.1 s, gradient 3.0 s), about 19 min for 200 epochs, against 908 s per epoch for the original
CPU code.
