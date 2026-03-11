# Profiling Memory
## Base Memory vs  Peak Memory
```
torch.cuda.reset_peak_memory_stats()
peak = torch.cuda.max_memory_allocated()
```

torch.cuda.max_memory_allocated() , gets the peak memory allocated since the last reset. So whatever memory was allocated before the reset , that is not accounted for

Example: Conside this code block , 

```
@contextlib.contextmanager
def torch_cuda_monitor():
    """Context manager to measure peak GPU memory in MB."""
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_alloc = torch.cuda.memory_allocated()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
            # return values indirectly by storing on the function object
            torch_cuda_monitor.peak_mb = peak_bytes / (1024**2)
            torch_cuda_monitor.start_mb = start_alloc / (1024**2)
    else:
        try:
            yield
        finally:
            torch_cuda_monitor.peak_mb = 0.0
            torch_cuda_monitor.start_mb = 0.0

```


Explanation 
``` 
# Step 0: which is designed to be a wrapper to be used like with torch_cuda_monitor
@contextlib.contextmanager
def torch_cuda_monitor():

......
.....

# Step 1: This gets the cuda memory allocated so far i.e. tensor etc. This could be 2000MB
# do a cuda synchornize as well
torch.cuda.synchronize()
start_alloc = torch.cuda.memory_allocated()

# Step 2: Reset the peak memory stats to zero
# Note : i) It only resets the peak statistics to zero
# ii) it does not release existing allocated memory . Hence the 2000MB is not released
torch.cuda.reset_peak_memory_stats()

# Step 3: Try yield: .
# Since this is a wrapper, the yield will be the part whatever code has to be profiled
# In the yield  block , peak memory is automatically tracked . The peak will be relative to the zero in Step 2
    try:
        yield

# Step 4: finally: .Finally block is executed just before the code exits
# Since max_memory_allocated was tracked, it measures the peak memory since the previous reset in Step2
# max_memory_allocate tracks only the peak memory for tensor allocation. It does not measure any cached blocks
# Hence torch.cuda.empty_cache() has no effect on torch.cuda.max_memory_allocated
# If the peak memory since step 2 was 1200 MB, then peak_bytes will hold 1200 MB (value will be stored in bytes though). 
# Total actual peak memory consumed at any point is 2000 +1200 = 2200 MB
# But the total memory used is 
    finally:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

# Step 5: reset PyTorch’s caching allocator between profiling runs
# emptying cache has got nothing to do with peak memory allocations
# since this is a wrapper and cache blocks continually get allocated, we want to release it before every 
# profiling run
        torch.cuda.empty_cache()

```

Contrast this with the piece of code from T2_memory_perplexity, demo
```
def compute_peak_memory_and_loss(model, inputs, device):
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs.get("attention_mask", None).to(device) if inputs.get("attention_mask") is not None else None,
            labels=inputs["labels"].to(device)
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mib = peak_bytes / 1024**2
    else:
        peak_mib = float("nan")
    return peak_mib, outputs.loss.item()

```

Explanation

Notice that compared to the torch_cuda_monitor
1) There is no torch.cuda.memory_allocated().: 
   - This is because compute_peak_memory_and_loss does not care about base memory. 
   - It only cares about peak memory. For peak memory only these two lines of code matter
```
torch.cuda.synchronize()
peak_bytes = torch.cuda.max_memory_allocated(device)
```

2) There is no torch.cuda.empty_cache
   - peak memory measures tensor allocations only. 
   -it does not measure cache blocks . so cuda.empty_cache has no bearing
   - compute_peak_memory_and_loss is not a profiler. Its only for one single run. we dont really care about clearing the cache. its an overkill

## Tensor Memory + Cache Memory
see tensor_memory_cache_memory.ipynb
Tensors:  memory_allocated()
1) torch.cuda.memory_allocated(): The amount of GPU memory currently used by active tensors.

Tensors + Cache ??: memory_reserved()
2) torch.cuda.memory_reserved(): The amount of GPU memory reserved by PyTorch's caching allocator.
   - ❌ It is not only cache memory. memory_reserved ≠ cache
   - ✅ It is total GPU memory held by PyTorch's caching allocator. memory_reserved = memory_allocated + cached_memory

