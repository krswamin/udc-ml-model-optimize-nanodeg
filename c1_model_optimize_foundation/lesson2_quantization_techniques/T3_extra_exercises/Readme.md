## References
- Refer to this video https://www.youtube.com/watch?v=0VdNflU08yA&t=830s
- Code inspired by / some version of the original code here : 
https://github.com/hkproj/quantization-notes/tree/main

### Full Static Post Training Quantization :  Summary
STEPS : full static Quantization
- i) Network : with input activation quantization
- ii)   specify a config: tells how to quantize weights and the activations
- iii)  fuse layers where possible like : linear and relu 
- iv) run calibration with representative data
- v)  quantize the model: convert the original weights to linear weights


```
STEP i) Network : with input activation quantization
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


STEP ii)   specify a config: tells how to quantize weights and the activations
# This does nothing to the model yet. It only sets the config
vsnet_quantized.qconfig = torch.ao.quantization.default_qconfig

STEP iii)  fuse layers where possible like : linear and relu (optional)
torch.ao.quantization.fuse_modules(
    vsnet_quantized, 
    [['linear1', 'relu1'], ['linear2', 'relu2']],
    inplace=True
)

STEP iv) 



