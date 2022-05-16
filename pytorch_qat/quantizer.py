from pytorch_qat.utils import *

import torch
import torch.nn as nn

from pytorch_qat.trainer import train_model


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


def quantize_aware_training(model, train_loader, test_loader, cfg, fuse_func = None):
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model.to(cuda_device)
    
    # ③ layer fusion을 적용합니다.
    # Fuse the model in place rather manually.
    if fuse_func == None:
        raise ValueError
    else:
        fused_model = fuse_func(model)

    # Print FP32 model.
    print(model)
    # Print fused model.
    print(fused_model)

    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model, model_2=fused_model, device=cuda_device,
    rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,3,32,32)), "Fused model is not equivalent to the original model!"

    
    quantized_model = QuantizedModel(model_fp32=fused_model)
    
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig
    # quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    quantized_model.qconfig = quantization_config
    
    print(quantized_model.qconfig)

    torch.quantization.prepare_qat(quantized_model, inplace=True)

    print("Training QAT Model...")
    quantized_model.train()
    
    train_model(model=quantized_model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, learning_rate=1e-3, num_epochs=10)
    
    quantized_model.to(cuda_device)
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    quantized_model.eval()

    # Print quantized model.
    print(quantized_model)

    # Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir=cfg["model_dir"], model_filename=cfg["quantized_model_filename"])

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath=cfg["quantized_model_filepath"], device=cuda_device)
    
    return quantized_model, quantized_jit_model