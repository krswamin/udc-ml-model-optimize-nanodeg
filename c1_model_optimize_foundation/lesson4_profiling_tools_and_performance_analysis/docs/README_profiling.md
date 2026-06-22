# PROFILING

```
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
```


- profile: main context manager for profiling. records cpu time, gpu/cuda time, memory usage
```
  with profile() as prof:
    model(x)
```
- ProfilerActivity: use it with profile to decide which hardware to profile on, cpu or gpu or both
```
with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ]
):
    model(x)
```
- schedule
- tensorboard_trace_handler: Saves profiler output so you can view it in TensorBoard
```
on_trace_ready=tensorboard_trace_handler("./log")
tensorboard --logdir=./log
```

final
```
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./log"),
    record_shapes=True
) as prof:

    for step in range(10):
        model(x)
        prof.step()
```
# KEY ANALYSIS METHODS
- PYNVML: analysis gpu hardware, high level (discussed in c3: efficient architecture, lesson1)
- PYTORCH PROFILER: analyze pytorch model when running training/inference
    - analyzes gpu kernel and cpu function call level
-TENSOR BOARD: visualize pytorch profiler results

# More on pytorch profiler, pynvml
- 💡☀️ This notebook has a lot of notes on pytorch profiler and tensor board: demo1_pytorch_profiler_ksw.ipynb
    - FULL PATH: udc-ml-model-optimize-nanodeg/c1_model_optimize_foundation/lesson4_profiling_tools_and_performance_analysis/T1_pytorch_profiler/demo1_pytorch_profiler_ksw.ipynb   
- 🌓 There is some code here: demo2_full_drill_profiling_ksw.ipynb
    - FULL PATH: udc-ml-model-optimize-nanodeg/c1_model_optimize_foundation/lesson4_profiling_tools_and_performance_analysis/T2_full_drill_profiling/demo2_full_drill_profiling_ksw.ipynb
- 🌓 Some minimal documentation here: README_profiling.md
    - FULL PATH: udc-ml-model-optimize-nanodeg/c1_model_optimize_foundation/lesson4_profiling_tools_and_performance_analysis/docs/README_profiling.md
- 💡☀️ A lot on pynvml and pytorch profiler in this readme :
    - FULL PATH: udc-ml-model-optimize-nanodeg/c3_efficient_architecture/lesson1_intro_hardware_aware_model_optimization/exercises/ksw_solution/README_performance_bottlenecks.md
- ☀️ Code with pyyvml and pytorch profiler here : exercise2_investigate_bottlenecks_ksw_soln.ipynb
    - FULL PATH: udc-ml-model-optimize-nanodeg/c3_efficient_architecture/lesson1_intro_hardware_aware_model_optimization/exercises/ksw_solution/exercise2_investigate_bottlenecks_ksw_soln.ipynb