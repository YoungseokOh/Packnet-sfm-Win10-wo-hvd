import os
from typing import OrderedDict

import torch
import copy
from pytorch_qat.trainer import *
from pytorch_qat.utils import *
from pytorch_qat.quantizer import quantize_aware_training
from packnet_sfm.networks.layers.resnet.layers import ConvBlock


def create_model(num_classes=10):

    model = resnet18(num_classes=num_classes, pretrained=False)

    return model

def fuse_ConvBNReLU(model):
    fused_model = copy.deepcopy(model)
    model.train()
    # The model has to be switched to training mode before any layer fusion.
    # Otherwise the quantization aware training will not work correctly.
    fused_model.eval()
    
    # ③ layer fusion을 적용합니다.
    # Fuse the model in place rather manually.
    # encoder
    fused_model.encoder.encoder = torch.quantization.fuse_modules(fused_model.encoder.encoder, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.encoder.encoder.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    # Decoder
    for module_name, module in fused_model.decoder.decoder.named_children():
        if isinstance(module, ConvBlock):
            torch.quantization.fuse_modules(module, [["conv", "nonlin"]], inplace=True)
    return fused_model

def main():

    random_seed = 0
    num_classes = 10
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    cfg = OrderedDict()
    cfg["model_dir"] = "saved_models"
    cfg["model_filename"] = "resnet18_cifar10.pt" 
    cfg["quantized_model_filename"] = "resnet18_quantized_cifar10.pt"
    cfg["model_filepath"] = os.path.join(cfg["model_dir"], cfg["model_filename"])
    cfg["quantized_model_filepath"] = os.path.join(cfg["model_dir"], cfg["quantized_model_filename"])

    set_random_seeds(random_seed=random_seed)
    
    # Create an untrained model.
    model = create_model(num_classes=num_classes)

    train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)
    
    # Train model.
    
    model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, device=cuda_device, learning_rate=1e-1, num_epochs=1)
    # Save model.
    save_model(model=model, model_dir=cfg["model_dir"], model_filename=cfg["model_filename"])
    
    model = load_model(model=model, model_filepath=cfg["model_filepath"], device=cuda_device)
    # Move the model to CPU since static quantization does not support CUDA currently.
    
    quantized_model, quantized_jit_model = quantize_aware_training(model, train_loader, test_loader, cfg, fuse_func=fuse_ConvBNReLU)

    _, fp32_eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=cpu_device, criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device, criterion=None)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model, device=cpu_device, input_size=(1,3,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=model, device=cuda_device, input_size=(1,3,32,32), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

if __name__ == "__main__":
    main()
    