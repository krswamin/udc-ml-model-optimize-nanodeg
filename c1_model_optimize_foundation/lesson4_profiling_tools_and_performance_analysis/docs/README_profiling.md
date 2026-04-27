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