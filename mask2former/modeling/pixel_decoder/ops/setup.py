# /home/luolu/PycharmProjects/Mask2FormerDinov2v3/mask2former/modeling/pixel_decoder/ops/setup.py
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension, CppExtension
import torch
import os

this_dir = Path(__file__).parent.resolve()
src_dir = this_dir / "src"

# 只放可编译的 .cpp / .cu；不要把 .h 写进 sources
sources_cpu = [
    str(src_dir / "cpu" / "ms_deform_attn_cpu.cpp"),
    str(src_dir / "ms_deform_attn.cpp"),           # << 新增绑定文件
]

sources_cuda = [
    str(src_dir / "cuda" / "ms_deform_attn_cuda.cu"),
]

def make_extensions():
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17",  # PyTorch 2.8 / GCC13
        ],
    }
    define_macros = []
    include_dirs = [str(src_dir)]

    if CUDA_HOME is not None and torch.cuda.is_available():
        extra_compile_args["nvcc"] = [
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-Xcompiler=-fPIC",
        ]
        # A6000：sm_86 + PTX
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6+PTX")

        return [
            CUDAExtension(
                name="ms_deform_attn",
                sources=sources_cpu + sources_cuda,
                include_dirs=include_dirs,
                define_macros=define_macros + [("WITH_CUDA", None)],
                extra_compile_args=extra_compile_args,
            )
        ]
    else:
        return [
            CppExtension(
                name="ms_deform_attn",
                sources=sources_cpu,
                include_dirs=include_dirs,
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        ]

ext_modules = make_extensions()

setup(
    name="ms_deform_attn",
    version="0.0.0",
    description="Multi-Scale Deformable Attention ops for Mask2Former (PyTorch 2.x compatible)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
