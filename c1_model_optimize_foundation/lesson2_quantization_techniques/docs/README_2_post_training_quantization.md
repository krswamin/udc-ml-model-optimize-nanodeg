# 🎯 C1, L2, T1 & T3: POST TRAINING QUANTIZATION
There are 3 types of quantizations
1) Weight only INT8 Quantization: 
   - Weights quantized to INT8, Activations remain FP32
2) Dynamic Quantization: 
   - Weights quantized to INT8, Activations become INT8 on the fly. But it takes time to convert them to INT8 since you have to analyze the min/max by looking at every batch. This increares RUN TIME INFERENCE MEMORY and reduces RUN TIME INFERENCE SPEED
3) Full Static Quantization
    - Weights quantized to INT8, Activations become INT8 on the fly. Activation INT8 conversion happens quickly because a calibration dataset was already used

![alt text](readme_imgs/ptq1_high_level_summary.png)


## 🎯 1. WEIGHTS ONLY INT8 QUANTIZATION

- WEIGHTS: parameter weights are quantized offline to INT8
- ACTIVATIONS: DO NOT get quantized . Activations remain FP32 . THEY ARE NOT QUANTIZED EVEN AT RUN TIME . 
- 🙂 COMPLEXITY/ NO CALIBRATION: 
  - Since activations are not being quantized, no calibration is needed. 
  - No Calibration: hence pretty stable, pretty Easy to Implement
- 🙂 STORAGE MEMORY: 
  - weights have gone from FP32 to INT8. Memory goes down by 4 times. 
  - activations are not quantized. They are FP32. No savings in memory there. 
  - Even IF ACTIVATIONS ARE QUANTIZED , THEY ARE NOT STORED. THEY DO NOT AFFECT STORED MODEL SIZE (see DETAILS D1 : STORAGE MEMORY ON DISK chart below ) 
  - but despite the activations in FP32, weights are the bulk of any model size. Significant model storage savings nevertheless
- RUNTIME INFERENCE : MEMORY, SPEED, ACCURACY
  - During Runtime the weights are INT8, the activations are FP32. But Matmul happens in FP32. So the weights need to be scaled. 
  - MATMUL: ~= (Weights_INT8* Scale)*Activation_FP32: This matmul is not as efficient as when both the weights and activations are INT8, But it is still better than Weights and Activation are both in FP32
  - 😟 RUNTIME INFERENCE MEMORY:  Highest Requirement of the 3. (See DETAILS D2 : RUNTIME INFERENCE MEMORY chart below)
  - 😟 RUNTIME INFERENCE SPEED: Lowest of the 3, since activations are FP32 and matmul is done in FP32
  - 🙂 RUNTIME INFERENCE ACCURACY: Highest of the 3, since activations are FP32 and matmul happens in FP32

\
\


##  ☀️ DETAILS
## ☀️ DETAILS D1 : MODEL STORAGE MEMORY 
- From the below notice that all 3 models when stored in CPU/HARD DISK. They virtually have the same foot print
- Hence they would have the same footprint when loading onto the VRAM (GPU) as well
- Therefore at this point the method of quantization, whether it is for a high accuracy application, or low memory application such as edge device , a lot of it is determined by the  i) Model Run Time (Inference) Performance ii) Ease of Quantization (and not on Model Storage Memory)

```
--------------------------------------------------------
Storage Memory (Model on Disk)
--------------------------------------------------------

0. Baseline FP32 (No Quantization)
Weights      : 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩  2 GB
Activations  : 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨  2 GB (not stored)
Total        : 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 ≈ 2 GB stored (activations not stored)

1. Weight-only INT8
Weights      : 🟩🟩🟩🟩  0.5 GB
Activations  : 🟨        (not stored)
Total        : 🟦🟦🟦🟦  ≈ 0.5 GB stored (activations not stored)


2. Dynamic INT8
Weights      : 🟩🟩🟩🟩  0.5 GB
Activations  : 🟨        (not stored)  
Total        : 🟦🟦🟦🟦  ≈ 0.5 GB stored (activations not stored)

3. Full/Static INT8
Weights      : 🟩🟩🟩🟩  0.5 GB
Activations  : 🟨        (not stored)  
Total        : 🟦🟦🟦🟦  ≈ 0.5 GB stored (activations not stored)
```

## ☀️ DETAILS D2 : MODEL RUNTIME INFERENCE MEMORY 
Runtime Inference Memory , Speed and Accuracy is one of the biggest drivers for model quantization choice
-  High Accuracy, but lower speed and  higher inference memory okay: Use Weights Only Quantization
-  High Speed is absolute requirement: Use Full Static Quantization (But need to go through the pain of getting an accurate calibration dataset)
- Somewhere in between but dont want to calibrate and stuff : Use Dynamic Quantization
```
--------------------------------------------------------
Memory Usage During Inference (Runtime)
--------------------------------------------------------

0. Baseline FP32 (No Quantization)
Weights      : 🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩  2 GB
Activations  : 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨  2 GB
Total        : 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 ≈ 4 GB

1. Weight-only INT8
Weights      : 🟩🟩🟩🟩  0.5 GB
Activations  : 🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨🟨  2 GB
Total        : 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 ≈ 2.5 GB

2. Dynamic INT8
Weights      : 🟩🟩🟩🟩  0.5 GB
Activations  : 🟨🟨🟨🟨🟨🟨🟨🟨  ~1.0 GB
Total        : 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦  ≈ 1.5 GB
Note: Activations quantized on-the-fly; some FP32 intermediates may exist.

3. Full/Static INT8
Weights      : 🟩🟩🟩🟩  0.5 GB
Activations  : 🟨🟨🟨🟨  0.5 GB
Total        : 🟦🟦🟦🟦🟦🟦🟦🟦  ≈ 1.0 GB
Note: Activations fully INT8 → maximum runtime memory reduction.
```

## ☀️ DETAILS D1 : QUANT & DEQUANT INSERTION
- DEQUANT inserted where necessary
  - In the Matmul Weights_INT8 are scaled to make it floating point. Conversion of INT to Floating Point is called DEquantization. This is the layer that is inserted on the fly between the weights and activation to be able to do multiplication in FP32. 
    ```
        FP32 activation x
            │
            ▼
        DeQuant layer: W_int8 → W_fp32
            │
            ▼
        MatMul (FP32): x_fp32 @ W_fp32
            │
            ▼
        FP32 output activation
    ```
- QUANT inserted where necessary
    - Sometimes a Floating point needs to be converted to INT. This is called Quantization (lol thats what we are studying too !) The layer that does it is called QUANT LAYER. 
    - Example: In a model where both the weights and activations are quantized. You might need to convert the output from  could be converting the final output from INT8 to floating point. This will need a Quant Layer
- NOT EVERYTHING CAN RUN INT8
  - Realistically not everything can run in INT8, some run on FP16/FP32. In this case Quant and DeQuant are inserted accordingly
  ```
    | Operation       | Usually INT8?   |
    | --------------- | --------------- |
    | Matrix multiply | Yes             |
    | LayerNorm       | Often FP16/FP32 |
    | Softmax         | FP16/FP32       |
    | GELU            | FP16/FP32       |

  ```
    