// ms_deform_attn.cpp -- minimal pybind11 bindings for the extension
#include <torch/extension.h>
#include "ms_deform_attn.h"

// Forward & Backward are declared in ms_deform_attn.h and implemented in cpu/cuda sources.
// They both return std::vector<at::Tensor>.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "ms_deform_attn_forward",
      &ms_deform_attn_forward,
      "MS Deformable Attention forward (CPU/CUDA)");

  m.def(
      "ms_deform_attn_backward",
      &ms_deform_attn_backward,
      "MS Deformable Attention backward (CPU/CUDA)");
}
