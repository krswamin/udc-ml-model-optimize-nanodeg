## Topics
1) Training: Quantization Aware Training. To recover accuracy lost due to quantization
   - During the training itself  weave quantization awareness into the training. Integrate low bit constraints into training.
   Tools 
    - Hugging Face Optimum
    - ONNX Runtime
    - bitsandbytes
2) POST TRAINING : Quantization
   - Convert full precision weights to lower precision weights. A new model emerges in minutes. \ 
     Example: Convert a FP32 trained model to INT8. Quick offline conversion
   - Dramatically compresses model and reduces size
   - Faster Inference
   - But keeps(should keep) the performance roughly the same


# Meaning of Quantization
- rounding 2.976 in a grocery store to 3 makes it easier to deal with
- similarly the value of pi
  - float32 (7 decimal digits)  : 3.1415927
  - float16 (3-4 decimal digits): 3.140625 (its not 3.1416 because you can only respresent as multiples of powers of 2)
  - int8 and int4 could be 314 or so based on the scale used

## Memory: Storage, RAM, VRAM
🧠 Where does training happen: GPU or CPU
- Modern training is almost always on GPU. CPU training is very slow

🧠 What Happens During Training. Where is data stored
While training:
- Weights live in VRAM
- Gradients live in VRAM
- Optimizer states live in VRAM
- Computation happens on the GPU

But when you save a checkpoint:
- Weights are copied from VRAM → RAM
- Then written from RAM → SSD (storage)
- So checkpoints end up on disk.


## POST TRAINING QUANTIZATION
i) Choose scale and zero
Scale : is the least count of the ruler . A floating-point number that defines the size of a single step in the integer range. A smaller scale means higher precision. (KSW feel this Scale is a misnomer for this, because Scale sounds like Range, but it is actually the least count)
ii) Post Training Quantization: 