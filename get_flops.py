import torch
import argparse
import numpy as np
# import fvcore.nn.flop_count as flop_count
from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
from fvcore.nn.jit_analysis import _IGNORED_OPS
from fvcore.nn.jit_handles import get_shape, addmm_flop_jit
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoModelForCausalLM, AutoConfig

# fvcore.nn.jit_handles
def get_flops_einsum(input_shapes, equation):
    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            # divided by 2 because we count MAC (multiply-add counted as one flop)
            flop = float(np.floor(float(line.split(":")[-1]) / 2))
            return flop

def flops_layernorm_ref(inputs, outputs):
    B, L, D = inputs[0].type().sizes()
    return B * L * D * 2

def flops_mamba_inner_ref(inputs, outputs):
    """
    Computes the approximate FLOPs for the mamba_inner_ref function.

    Parameters:
    B (int): Batch size
    L (int): Sequence length
    D (int): Model Proj Dimension (d_model * expand)
    N (int): State dimension
    K (int): Kernel size of the 1D convolution
    delta_rank (int): Rank for delta projection
    d_model (int): Hidden dimension
    """

    B, _, L = inputs[0].type().sizes()
    D, _, K = inputs[1].type().sizes()
    dbl = inputs[3].type().sizes()[0]
    dt_rank = inputs[4].type().sizes()[1]
    N = inputs[6].type().sizes()[1]
    d_model = outputs[0].type().sizes()[2]
    padding = 8

    conv1D_flops = B * D * L * K # conv1d
    print("conv1D: ".ljust(padding), conv1D_flops)

    x_proj_flops = get_flops_einsum([[B, L, D], [D, dbl]], "bld,dr->blr") # x_proj
    print("x_proj: ".ljust(padding), x_proj_flops)

    delta_flops = get_flops_einsum([[B, L, dt_rank], [dt_rank, D]], "bld,dr->blr") # delta_proj
    print("delta: ".ljust(padding), delta_flops)

    scan_flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True, with_complex=False)
    print("scan: ".ljust(padding), scan_flops)

    out_flops = get_flops_einsum([[B, L, D], [D, d_model]], "bld,dr->blr") # out_proj
    print("out: ".ljust(padding), out_flops)

    flops = conv1D_flops + x_proj_flops + delta_flops + scan_flops + out_flops

    in_flops = get_flops_einsum([[B, L, d_model], [d_model, D * 2]], "bld,dr->blr") # in_proj
    print("in: ".ljust(padding), in_flops)

    return flops

def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")  # deltaA
    if with_Group:  # deltab_u
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N    # x = deltaA[:, i] * x + deltaB_u[:, i]
    if with_Group:  # y = x * c
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L

    return flops


parser = argparse.ArgumentParser()
# 1.facebook/opt-125m
# 2.state-spaces/mamba-130m
parser.add_argument("--model", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--seq_len", type=int, default=2048)
args = parser.parse_args()

if "opt" in args.model:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
else:
    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.float16, device="cuda")

supported_ops={
    "aten::silu": None,
    "aten::neg": None,
    "aten::exp": None,
    "aten::flip": None,
    "prim::PythonOp.MambaInnerFn": flops_mamba_inner_ref,
    "prim::PythonOp.LayerNormFn": flops_layernorm_ref,
}

input_shape = (args.batch, args.seq_len)
inputs = torch.randint(1, 1000, (args.batch, args.seq_len), dtype=torch.long, device="cuda")

model.eval()
Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)

flops_table = flop_count_table(
    flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
    max_depth=100,
    activations=None,
    show_param_shapes=True,
)

flops_str = flop_count_str(
    flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
    activations=None,
)

print(flops_str)
print(flops_table)
params = fvcore_parameter_count(model)[""]
flops = sum(Gflops.values())
print("GFlops: ", flops, "Params: ", params, flush=True)