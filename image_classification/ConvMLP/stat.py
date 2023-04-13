import os
import glob
import paddle
from config import get_config
from convmlp import build_convmlp as build_model

def count_gelu(layer, inputs, output):
    activation_flops = 8
    x = inputs[0]
    num = x.numel()
    layer.total_ops += num * activation_flops 


def count_softmax(layer, inputs, output):
    softmax_flops = 5 # max/substract, exp, sum, divide
    x = inputs[0]
    num = x.numel()
    layer.total_ops += num * softmax_flops 


def count_layernorm(layer, inputs, output):
    layer_norm_flops = 5 # get mean (sum), get variance (square and sum), scale(multiply)
    x = inputs[0]
    num = x.numel()
    layer.total_ops += num * layer_norm_flops 


for cfg in glob.glob('./configs/*.yaml'):
    #cfg = './configs/pvtv2_b0.yaml'
    #input_size = (1, 3, 512, 512)
    #input_size = (1, 3, 448, 448)
    #input_size = (1, 3, 384, 384)
    #input_size = (1, 3, 256, 256)
    input_size = (1, 3, 224, 224)
    config = get_config(cfg)
    model = build_model(config)
    
    custom_ops = {paddle.nn.GELU: count_gelu,
                  paddle.nn.LayerNorm: count_layernorm,
                  paddle.nn.Softmax: count_softmax,
                }
    print(os.path.basename(cfg))
    paddle.flops(model,
                 input_size=input_size,
                 custom_ops=custom_ops,
                 print_detail=False)


#for cfg in glob.glob('./configs/*.yaml'):
#    #cfg = './configs/swin_base_patch4_window7_224.yaml'
#    input_size = (1, 3, int(cfg[-8:-5]), int(cfg[-8:-5]))
#    config = get_config(cfg)
#    model = build_model(config)
#    
#    
#    custom_ops = {paddle.nn.GELU: count_gelu,
#                  paddle.nn.LayerNorm: count_layernorm,
#                  paddle.nn.Softmax: count_softmax,
#                }
#    print(os.path.basename(cfg))
#    paddle.flops(model,
#                 input_size=input_size,
#                 custom_ops=custom_ops,
#                 print_detail=False)
#    print('-----------')