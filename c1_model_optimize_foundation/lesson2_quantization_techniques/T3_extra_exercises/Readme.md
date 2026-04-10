## References & Citations
- Refer to this video UMAR JAMIL: https://www.youtube.com/watch?v=0VdNflU08yA&t=830s
- Code inspired by the original code here : 
https://github.com/hkproj/quantization-notes/tree/main

### Full Static Post Training Quantization :  Summary
This 
- quantized the inputs, weight and activation
- uses torch.ao.quantization
- requires calibration

STEPS : full static Quantization
- i)   Import the necessary libraries
- ii)   Network : with input activation quantization
- iii)  specify a config: tells how to quantize weights and the activations
- iv) fuse layers where possible like : linear and relu 
- v)  insert min max observers in the model
- vi)   run calibration with representative data
- vii)   quantize the model: using the calibration statistics
- viii)  compare original and quantized model size
- ix) Dequantize weights and see how that looks




```
# STEP i) Import the necessary libraries
# We will use pytorch's quantization library torch.ao.quantization
import torch

# STEP ii) Network : with input activation quantization
class QuantizedVerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(QuantizedVerySimpleNet,self).__init__()
        self.quantize = torch.quantization.QuantStub()
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.dequantize = torch.quantization.DeQuantStub()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.quantize(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.dequantize(x)
        return x

# Instantiate Quantization ready model
vsnet_quantized = QuantizedVerySimpleNet().to(device)
# Copy weights from unquantized model
vsnet_quantized.load_state_dict(vsnet.state_dict())
# Put the layers in eval mode (i.e. no dropout, batch norm uses stored statistics) 
vsnet_quantized.eval()


# STEP iii) Specify a config: tells how to quantize weights and the activations
# This does nothing to the model yet. It only sets the config
vsnet_quantized.qconfig = torch.ao.quantization.default_qconfig

# STEP iv) Fuse layers where possible like : linear and relu (optional)
torch.ao.quantization.fuse_modules(
    vsnet_quantized, 
    [['linear1', 'relu1'], ['linear2', 'relu2']],
    inplace=True
)

# STEP v) Insert Min-Max Obserers in the model
vsnet_quantized = torch.ao.quantization.prepare(vsnet_quantized)

# STEP vi) Calibrate the Model using Test Set
test(vsnet_quantized)

# STEP vii) Quantize the model
vsnet_quantized = torch.ao.quantization.convert(vsnet_quantized)

# STEP viii)  compare original and quantized model size
print_size_of_model(vsnet)
print_size_of_model(vsnet_quantized)

# STEP ix) Print weights, Dequantize weights and see how that looks
# print quantized INT weights
print(torch.int_repr(vsnet_quantized.linear1.weight()))
# print dequantized weights (back to float)
print(torch.dequantize(vsnet_quantized.linear1.weight()))
```



