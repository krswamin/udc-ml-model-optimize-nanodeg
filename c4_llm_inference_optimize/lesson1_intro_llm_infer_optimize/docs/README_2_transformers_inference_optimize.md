# 🎯 Transformer Inference Optimization
- Bottlenecks
- Goals
- Metrics
- Tools
- Solutions

## 🎯 1) Check Your Intuition
![alt text](readme_imgs/llm_inference_bottlenecks/llm_infer_check1.png) 
![alt text](readme_imgs/llm_inference_bottlenecks/llm_infer_check2.png) 
![alt text](readme_imgs/llm_inference_bottlenecks/llm_infer_check3.png) 

## 🎯 2) Goals : Metrics for Transformer Inference Optimization
Metrics
- Latency
- Throughput
- Memory/ Cost (not shown in pic)
- Perpelxity( this is not a metric for inference optimization apparently ???)
![alt text](readme_imgs/llm_inference_bottlenecks/llm_infer_goals.png) 
![alt text](readme_imgs/llm_inference_bottlenecks/llm_infer_latency.png) 
![alt text](readme_imgs/llm_inference_bottlenecks/llm_infer_throughput.png)

## 🎯 3) Bottle Necks
- attention mechanism : scales quadratically with sequence length
- auto regressive token generation : auto regressive decoding to generate tokens is sequential
- memory bandwidth limitations
![alt text](readme_imgs/llm_inference_bottlenecks/bottleneck0_overview.png) 

### 🎯 3.1) Attention Mechanism & Sequence Length
Attention Mechanism scales quadratically with sequence Length i.e O(N*N)
KSW TODO: paste the picture of handwritten notes
![alt text](readme_imgs/llm_inference_bottlenecks/bottleneck1_self_attention.png) 

### 🎯 3.2) Autoregresive Token Generation
Autoregressive decoding generates one token at a time. This makes the process inherently sequential
![alt text](readme_imgs/llm_inference_bottlenecks/bottleneck2_auto_regressive_token_generation.png) 


### 🎯 3.3) Memory Bandwidth Limitations
Limits how quickly you can deliver data to the compute units
![alt text](readme_imgs/llm_inference_bottlenecks/bottleneck3_memory.png)

## 🎯 4) Solutions
- **Transformer Architecture Transformations:** change things about the architecture 
see [lesson2_transformer_architecture_optimize/docs](../../lesson2_transformer_architecture_optimize/docs)
    - **KV Caching:** reduces the vast amount of redundant computations that happen due to step by step text generation
    - **GQA grouped query attention:** reduces the memory burden due to KV Caching
    - **Flash Attention:** Algorithm to speed up calculation by relying less on memory access. Speeds up core attention calculation
