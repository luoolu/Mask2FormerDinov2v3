# Compatibility shim so legacy imports keep working:
#   from .. import MultiScaleDeformableAttention as MSDA
# maps to the compiled extension module: ms_deform_attn
from . import ms_deform_attn as _ext

# expose the expected symbols
ms_deform_attn_forward = _ext.ms_deform_attn_forward
ms_deform_attn_backward = _ext.ms_deform_attn_backward

__all__ = ['ms_deform_attn_forward', 'ms_deform_attn_backward']
