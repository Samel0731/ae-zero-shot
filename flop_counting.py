from model import build_AE
import torch

def meta_flop_counter():
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    model = build_AE(encoder_type='convnext')
    batch_size = 1
    input_shape = (batch_size, 1, 200, 200)
    input_tensor = torch.randn(input_shape)

    flops = FlopCountAnalysis(model, input_tensor)
    print(flops.total())
    print(flops.by_operator())
    print(flops.by_module())
    print(flop_count_table(flops))

def thop_flop_counter():
    from thop import profile
    from thop import clever_format

    model = build_AE(encoder_type='convnext')
    batch_size = 1
    input_shape = (batch_size, 1, 200, 200)
    input_tensor = torch.randn(input_shape)

    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

def ptflop_flop_counter():
    from ptflops import get_model_complexity_info

    model = build_AE(encoder_type='convnext')
    input_shape = (1, 200, 200)
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False, verbose=True)
    print(macs, params)

def deepspeed_flop_counter():
    from deepspeed.profiling.flops_profiler import get_model_profile

    model = build_AE(encoder_type='convnext')
    batch_size = 1
    input_shape = (batch_size, 1, 200, 200)
    flops, macs, params = get_model_profile(model, input_shape)
    print(flops, macs, params)  # 1.02 G 508.79 MMACs 138.48 K

if __name__ == '__main__':
    deepspeed_flop_counter()