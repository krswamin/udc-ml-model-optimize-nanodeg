# MODEL PERFORMANCE BOTTLE NECKS

# 🎯 1) INEFFICIENT NEURAL NETWORKS
## 🎯 1.1)  FULLY CONNECTED LAYERS
These add the most number of parameters even in transformer architecture
To reduce the number of parameters
 - reduce the number of hidden layers
 - reduce the number of neurons in each layer

```
# Example 1: the input is a tensor. but the dataset is just a csv file
        for hidden_size in hidden_sizes:

            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
```

The number of fully connected layers  is an overkill for a csv.  \
You probably dont need the hidden layers to be 512 or 1024 either. thats too big. \

Number of parameters
    - i) input_size* 512: [input_size]*512
    - ii)   512*1024 = 524288 parameters
    - iii) 1024*512  = 524288 parameters
    - iv)   512*256  = 131072 parameters
    - v)    256*128  =  32768 parameters
Total Parameters ~= 1.2 Million + [input_size]*512 atleast

```
# Example 2
class InefficientCNN(nn.Module):
    """
    Intentionally inefficient CNN for performance profiling practice.

    Common real-world issues included:
    1. Too many small layers
    2. Redundant operations
    3. Unnecessary tensor copies
    4. Bad memory layout changes
    5. Overuse of Python loops
    6. Repeated activation functions
    7. Unnecessary normalization
    8. Large fully connected layer
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        # Intentionally large fully connected layer
        self.fc1 = nn.Linear(32 * 224 * 224, 512)
        self.fc2 = nn.Linear(512, num_classes)
```

Number of parameters: 
- fc1 : (32 * 224 * 224 )* 512 = (1605632 * 512) = 822 million parameters
   - **WOW 822 Million. Let that sink in:** 2,224,224 seem harmless. combined that itself is 1.6 million
- fc2: 512*num_classes = 512 *10 = 5120
- total parameters = 822 million parameters. See the mu
            

## 🎯 1.2) CREATING UNNECESSARY TENSORS
```
# Extra tensor operations that fragment GPU utilization
x = x + 0.0001  # Unnecessary operation that creates new tensor
x = x * 1.0001  # Another unnecessary operation that creates another tensor

Tensor x1
   ↓ add kernel, aten::add
Tensor x2
   ↓ multiply kernel, aten::mul
Tensor x3
 even though they all go by x. These are 3 different tensors
```

- In pytorch by default all operations are out of place/ they are not in place. This is because a forward graph needs to be built and two tensor being simultaneously present i.e the original-x tensor and the new-x tensor is needed for back propagation (???)

- In the of place operations a whole bunch of things are happening
    - so x = x + 0.0001 is actually two Xes. x2 = x1 + 0.0001. x1 and x2 are two different tensors
        - The following steps happen in an out of place addition
        - Allocate memory for tensor x2
        - read x1 from memory
        - Copy x1 to x2. i.e write into x2
        - launch the **ADDITION GPU KERNEL aten::add** & add 0.0001 to x2
        - Reassign Python variable x

    - when you do another addition x = x * 1.0001 . These are two different Xes as well. x3 = x2 + 1.0001. 
       - all the above operations are triggered again  (except a **MULTIPLY GPU KERNEL , aten::mul** is triggered instead of ADDITION GPU KERNEL)  

    - x1, x2, x3 are all new tensors. Do they all coexist in memory ? lets walk through it
      - at first x1 and x2 coexist in memory
      - When adding x2 to x3, x1 might or might not exist based on the following
        - Autograd may need it: Autograd may keep x1 and x2 alive until loss.backward() finishes (Autograd is PyTorch's automatic differentiation engine. It automatically compute gradients for training neural networks, during back propagation.)
        - so if autograd is the reason, and lets in inference mode autograd (back propagation) has been turned off. Then x1 might not exist (provided no other python references exist and cuda kerel competed etc.)
        - Other Python references may exist
        - The CUDA kernel may not have completed yet


### 🎯 1.2.1) Are in place Memory Operations an option ?
- Pytorch offers in place memory operations
    ```
    x.add_(0.0001)
    x.mul_(1.0001)


    Tensor x
    ↓ modify in-place
    Tensor x
    ↓ modify in-place
    Tensor x
    ```
- but this might break autograd which expects original tensors in place to do the autgrad
    ```
    RuntimeError:
    one of the variables needed for gradient computation
    has been modified by an inplace operation
    ```
Hence out of place operations better with pytorch



## 🎯 1.3) GPU-CPU SYNCHRONIZATION POINTS ARE BAD !

```
    def forward(self, x):
        
        # 6. Inefficient forward pass with unnecessary operations
       
        batch_size = x.size(0)        
        # Extra tensor operations that fragment GPU utilization
        x = x + 0.0001  # Unnecessary operation that creates new tensor
        x = x * 1.0001  # Another unnecessary operation
        
        # Pass through main network
        x = self.network(x)
        
        # 7. Inefficient final processing
        # Force CPU-GPU sync (performance killer!)
        if batch_size > 1:
            # This creates a CPU-GPU synchronization point
            # This is the absolute worst.
            # x is on the GPU. torch.mean(x) computes a GPU tensor. But .item() asks Python on the CPU to get the scalar value
            # the final variable mean_activation lives on the CPU. It is a CPU-side Python scalar.
            mean_activation = torch.mean(x).item()  # .item() forces sync!
            
        return x
```
- mean_activation = torch.mean(x).item()  # .item() forces gpu-cpu sync!
    - torch.mean(x), if x is on GPU, then torch.mean(x) launches a GPU reduction kernel and returns a PyTorch tensor:
    - but .item() .  can happen only on the CPU. .item() is actually also a PyTorch method, not a Python , but it returns a native Python scalar. .item() is not a computation. It's an extraction of a value from a tensor into Python.
        ```
        # Consider the following example
        y = torch.tensor(3.14, device="cuda"). # y is a tensor on the GPU

        # the final variable v lives on the CPU. It is a CPU-side Python scalar.
        v = y.item()                           # v is a float on the CPU

        print("v =",v, "type-v=", type(v))     # prints:  v = 3.14, type-v = float
        ````
        - so v is a float. Floats can only exist in the python interpreter. Python interpreter is on the CPU

    - GPU and CPU are ascynchronous. CPU triggers tasks on GPU and then CPU and GPU do their own thing independently. But in this case
      - GPU must finish computing the mean
      - Result must be copied GPU → CPU
      - Python float must be created


# 🎯 2) pynvml 
- pynvml is a Python wrapper around NVML (NVIDIA Management Library).
- It allows Python programs to query GPU status without launching tools like nvidia-smi.
- NVML is a C library provided by NVIDIA.The same library is used internally by: nvidia-smi
- nvidia-smi outputs the following. Most of that information comes from NVML.
    - GPU Utilization
    - Memory Usage
    - Temperature
    - Power
    - Clocks
    
Think of it like this:
    ```
    Python Code
        ↓
    pynvml
        ↓
    NVML Library
        ↓
    NVIDIA Driver
        ↓
    GPU
    ```

GPU has two sides. Its useful to keep this distinction to understand which pnvml function works on which side
        ```
        Compute Side                    Memory Side

        CUDA Cores                      GDDR6 VRAM
        Tensor Cores                    Memory Controllers
        FP32 Units                      Memory Bus
        SMs                             L2 Cache
        ```

## 🎯 2.1) pnvml functions: GPU Utilization, Memory: Compute vs Memory


| Metric         | Absolute or % | Example |
| -------------- | ------------- | ------- |
| `util.gpu`     | Percentage    | 87 (%)  |
| `util.memory`  | Percentage    | 42 (%)  |
| `memory.used`  | Absolute      | 3.2 GB  |
| `memory.free`  | Absolute      | 4.8 GB  |
| `memory.total` | Absolute      | 8.0 GB  |
| `power_usage`  | Absolute      | 52 W    |
| `power_limit`  | Absolute      | 80 W    |
| `temp`         | Absolute      | 76 °C   |


**The Most Nuanced -1**
- **GPU Utilization:** util = nvmlDeviceGetUtilizationRates(handle)
    - util.gpu = 87. this is a percentage
        - Gpu compute busy 87% of the time doing computations like addition, multiplications, etc.
        - NVML gives the high-level "GPU busy %" number, not the breakdown of which execution units are doing the work.  no break down of gpu cores , tensor cores, rt cores etc.
    - util.memory = 42. this is a percentage
        - util.memory is not GPU memory/VRAM. It refers to the Memory Controller(Memory traffic / bandwidth utilization). util.memory = 42 means that the Memory controller busy 42% of time. This is like traffic on the highway. how many reads/ writes to and from memory are happening
- **GPU Memory:** memory = nvmlDeviceGetMemoryInfo(handle)
    - This is the actual GPU VRAM Memory (this is not the memory traffic. Think of it as a parking lot where potentially many cars could be parked)
    - memory.used	How much VRAM is allocated. this is an absolute value
    - memory.free	Remaining VRAM. This is an absolute value
    - memory.total	Total VRAM capacity. This is an absolute value



```
Examples: contrived by KSW.  dont get into the nitty-gritties. follow along the main idea . LOL !

# Example 1: GPU is idle
util.gpu = 0
util.memory = 0
memory.used = 0
memory.free = 16
memory.total = 16

# Example 2a: Large Model loaded into VRAM. No training or inference yet
# no read or write, hence util.memory  = 0 %
# No computation yet, hence util.gpu = 0%
# memory.used = 12 gb is the volume used by the model on the VRAM
util.gpu = 0
util.memory = 0
memory.used = 12
memory.free = 4
memory.total = 16

# Example 2b: Large Model loaded into VRAM. Training Running
# Lots of read/write traffic hence util.memory is 50 %
# Lots of computation in the gpu, hence util.gpu = 80% 
# memory.used = 12 gb is the volume used by the model on the VRAM, additional 3gb because tensors are being created etc during training
util.gpu = 80
util.memory = 50
memory.used = 15
memory.free = 1
memory.total = 16

# Example 2c: Large Model loaded into VRAM. Inference Running,
# throughput maximized for inference, but model size reduced for inference to int8
# Lots of read/write traffic of the model optimized for throughput hence util.memory is 60 %, but we are still not hitting any memory bandwidth blocks. this is good
# Lots of computation in the gpu, gpu maximized for computation hence util.gpu = 90% 
# memory.used = 5 gb . 12gb fp32 models becomes int8. So the model is now 12/4 = 3gb.  (much lower than used by the fp32 model), additional 2gb is  the memory used by the tensors that are being created etc during inference
util.gpu = 90
util.memory = 60
memory.used = 6
memory.free = 10
memory.total = 16

# Example 3a: Small Model loaded into VRAM. No training or inference yet
util.gpu = 0
util.memory = 0
memory.used = 6
memory.free = 10
memory.total = 16

# Example 3b: Small Model loaded into VRAM. Training Running, 
util.gpu = 20
util.memory = 20
memory.used = 6
memory.free = 10
memory.total = 16

# Example 3c: Small Model loaded into VRAM. Inference Running, 
# throughput maximized for inference, but model size reduced for inference to int8
# memory.used = 2.5 gb . 6gb fp32 models becomes int8. So the model is now 6/4 = 1.5gb.  (much lower than used by the fp32 model), additional 1gb is  the memory used by the tensors that are being created etc during inference
util.gpu = 80
util.memory = 80
memory.used = 2.5
memory.free = 13.5
memory.total = 16
```
| util.gpu | util.memory      | Meaning                                     |
| -------- | ---------------- | ------------------------------------------- |
| High     | Low              | Compute-bound                               |
| Low      | High             | Memory-bound                                |
| High     | High             | Well-utilized workload                      |
| Low      | Low              | CPU bottleneck / idle                       |
| High     | VRAM almost full | Capacity limit approaching                  |
| Low      | VRAM almost full | Large model loaded but not actively running |

## 🎯 2.2) pnvml functions: GPU Clocks: Compute vs. Memory
- **Graphics Clock/ GPU Compute Clock:** 
    - graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    - This returns the current frequency (MHz) of the GPU's compute domain.
    - This is the clock that drives the GPU compute core (so includes GPU Cores, Tensor cores , FP32 units, INT8 units,  Scheduling logic, Most compute pipelines everything)
    - This is the clock that controls/ is controlled by the compute load
    - Larger the load higher the frequency of the clock(recall computer architecture design in UMICH where the clock rate controlled performance). So does the clock frequency control the load or the load drives the clock frequency higher ?
    - Does the clock control the load or does the load drive the clock?
      - answer: Both, but the primary direction is: Workload → Clock Frequency
      - The GPU continuously monitors: Utilization, Power consumption, Temperature, Voltage limits and adjusts clocks accordingly.
            ```
            More load
                ↓
            Need more performance
                ↓
            GPU boosts clock
            ```

- **GPU Memory Clock:** 
    - memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    - This is the GPU Memory clock / GPU VRAM
    - Controls how fast data moves between VRAM, Memory controllers, L2 Cache, Tensor cores etc. Higher memory clock ⇒ more bytes/sec.
            ```
            VRAM
            ↕
            Memory Controllers
            ↕
            L2 Cache
            ↕
            SMs / Tensor Cores
            ```





## 🎯 2.3) pnvml functions: GPU Performance 
- **Performance State:** perf_state = pynvml.nvmlDeviceGetPerformanceState(handle)
  - how does this really measure performance ??? , not sure
  - is it speed, accuracy or like what ??
- **Throttle Reason:**
    - throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)   
    - throttle measn GPU downgrades the speed so that it doesnt get overheated/ or to conserve battery power
    -this function outputs a number. each number corresponds to a code which tells why the GPU is throttling. It could be because something is blocking the gpu fan, or compute is saturated or memory is saturated
    - if it is not throttling the output correcponds to 'NONE" or "No throttle"


## 🎯 2.4) pnvml functions: Power & Temperature
- **GPU Temperature:** temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    -  this is usually the GPU Core Temperature
    -  other temperatures such as GPU VRAM Temperature, Gpu Hotspot Temperature can be obtained based on the GPU version, pynvml version
    -  nvidia-smi shows NVML_TEMPERATURE_GPU, which is the GPU Core Temperature
- **GPU Power:** power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    - how much power the GPU is currently using. This is the power consumed by VRAM (the storage), tensor cores, cuda cores, (both are very compute occurs), memory controller ( the traffic bandwidth )
- **GPU Power Limit:** pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    - maximum power gpu is allowed to use. Exampl 40W. This really varies widely if this is a Laptop, A100, H100, or Nvidia-Drive-Thor
    ```
    # Example
    
    # Idle GPU:
    GPU Utilization = 0%
    Power Usage = 8 W
    
    # Inference:
    GPU Utilization = 70%
    Power Usage = 45 W
    
    # Training:
    GPU Utilization = 98%
    Power Usage = 78 W
   ```


## 🎯 2.5) Typical Tesla T4 Summary: Example of pnvml functions in use
| Workload            | GPU Util | Graphics Clock | Memory Clock |
| ------------------- | -------- | -------------- | ------------ |
| Idle                | 0%       | 300 MHz        | 405 MHz      |
| Light Inference     | 10-20%   | 600-900 MHz    | 5000 MHz     |
| Real-Time Inference | 60-90%   | 1200-1500 MHz  | 5000 MHz     |
| Heavy Training      | 95-100%  | 1500-1590 MHz  | 5000 MHz     |
| Memory-Bound        | 20-40%   | 800-1000 MHz   | 5000 MHz     |
| Thermal Throttled   | 100%     | 1000-1300 MHz  | 5000 MHz     |
- For NVIDIA datacenter GPUs like the NVIDIA Tesla T4, the memory clock is often fixed near its maximum (≈5000 MHz) once the GPU is doing meaningful work.
- The graphics clock is the one that changes most dramatically with workload, temperature, and power headroom. 
- That's why, when profiling AI workloads, engineers usually watch the following. Together—they tell you whether the GPU is idle, fully utilized, or being throttled.
    - util.gpu
    - graphics_clock
    - temperature
    - power_usage


# 🎯 3) ynvml vs pytorch profiler
## 🎯 3.1) pynvml vs pytorch profiler: FUNDAMENTAL DIFFERENCE
- pynvml is at a higher GPU level. percetage gpu utilization, gpu temperature, gpu clock speeds etc
    - pynvml queries the hardware state . You can do this when code is running, when gpu is idle etc. at anytimg
- pytorch profiler is DEEPER. It captures GPU is at the kernel level. It can capture both CPU and GPU though, at a deeper level
    - pytorch captures data on a software execution stack- specifically on a Pytorch execution stack


## 🎯 3.2) pynvml vs pytorch profiler: CAN ONE REPLACE THE OTHER
Answer: NO. one cannot replace the other
pynvml cannot tell you the following
    - which layer/kernel is slow
    - Which operator allocates memory
    - nothing about the cpu processes
    - CPU vs CUDA breakdown
pytorch profiler
    - cannot tell temperature
    - cannot tell gpu utilization ?? really. what if you added all the memory ussage from the different kernels
    - cannot tell throttle reason


## 🎯 4) KEY ANALYSIS METHODS
- PYNVML: analysis gpu hardware, high level (discussed in c3: efficient architecture, lesson1)
- PYTORCH PROFILER: analyze pytorch model when running training/inference
    - analyzes gpu kernel and cpu function call level
-TENSOR BOARD: visualize pytorch profiler results

## 🎯 5) More on pytorch profiler, pynvml
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
