import contextlib
import threading
from typing import Any, Callable, Optional, Sequence

from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

_dtensor_singleton = None
_dtensor_singleton_lock = threading.Lock()

@tf_export("experimental.dtensor.call_with_layout", v1=[])
def call_with_layout(fn: Callable[..., Any], layout: Optional[layout_lib.Layout],
                     *args, **kwargs) -> Any:
    if layout is not None:
        if context.executing_eagerly():
            with default_mesh(layout.mesh):
                with _dtensor_device()._default_layout(layout):
                    return fn(*args, **kwargs)
        else:
            return relayout(fn(*args, **kwargs), layout)
    return fn(*args, **kwargs)

@tf_export("experimental.dtensor.run_on", v1=[])
@deprecation.deprecated(None, "Use `dtensor.default_mesh` scope instead.")
@contextlib.contextmanager
def run_on(mesh: layout_lib.Mesh):
    with default_mesh(mesh):
        yield

@tf_export("experimental.dtensor.default_mesh", v1=[])
@contextlib.contextmanager
def default_mesh(mesh: layout_lib.Mesh):
    if not isinstance(mesh, layout_lib.Mesh):
        raise ValueError(f"Expect `mesh` to be `Mesh`, got {type(mesh)}")

    with _dtensor_device()._experimental_default_mesh(mesh):
        with ops.device(device_name()):
            yield

@tf_export("experimental.dtensor.get_default_mesh", v1=[])
def get_default_mesh() -> Optional[layout_lib.Mesh]:
    if _dtensor_singleton is None:
        return None
    else:
        return _dtensor_singleton._current_default_mesh

@tf_export("experimental.dtensor.device_name", v1=[])
def device_name() -> str:
    return _dtensor_device().name

@tf_export("experimental.dtensor.is_dtensor", v1=[])
def is_dtensor(tensor) -> bool:
    return _dtensor_device().is_dtensor(tensor)

@tf_export("experimental.dtensor.copy_to_mesh", v1=[])
def copy_to_mesh(tensor: Any, layout: layout_lib.Layout,
                 source_layout: Optional[layout_lib.Layout] = None) -> tensor_lib.Tensor:
    del source_layout
    return relayout(tensor, layout)

@tf_export("experimental.dtensor.pack", v1=[])
def pack(tensors: Sequence[Any], layout: layout_lib.Layout) -> Any:
    return _dtensor_device().pack(tensors, layout)

@tf_export("experimental.dtensor.unpack", v1=[])
def unpack(tensor: Any) -> Sequence[Any]:
    return _dtensor_device().unpack(tensor)

@tf_export("experimental.dtensor.fetch_layout", v1=[])
def fetch_layout(tensor: tensor_lib.Tensor) -> layout_lib.Layout:
    return _dtensor_device().fetch_layout(tensor)

@tf_export("experimental.dtensor.check_layout", v1=[])
def check_layout(tensor: tensor_lib.Tensor, layout: layout_lib.Layout) -> None:
    if fetch_layout(tensor) != layout:
        raise ValueError("Layout of tensor: " + str(fetch_layout(tensor)) +
                         ", did not match expected layout: " + str(layout))

@tf_export("experimental.dtensor.relayout", v1=[])
def relayout(tensor: tensor_lib.Tensor, layout: layout_lib.Layout,
             name: Optional[str] = None) -> tensor_lib.Tensor:
    layout_str = layout.to_string()
    with default_mesh(layout.mesh):
        return gen_dtensor_ops.relayout(tensor, layout_str, name=name)

@tf_export("experimental.dtensor.relayout_like", v1=[])
def relayout_like(tensor: tensor_lib.Tensor, layout_tensor: tensor_lib.Tensor,
                  name: Optional[str] = None) -> tensor_lib.Tensor:
    return gen_dtensor_ops.relayout_like(input=tensor, layout_input=layout_tensor, name=name)

@tf_export("experimental.dtensor._reset_dtensor_device", v1=[])
def reset_dtensor_device(is_async: bool) -> None:
    global _dtensor_singleton
    device = dtensor_device.DTensorDevice(meshes=[], is_async=is_async)
    _dtensor_singleton = device

def _dtensor_device() -> dtensor_device.DTensorDevice:
    with _dtensor_singleton_lock:
        if _dtensor_singleton is None:
            reset_dtensor_device(is_async=True)
        return _dtensor_singleton

def _reset() -> None:
    global _dtensor_singleton
    with _dtensor_singleton_lock:
        if _dtensor_singleton is not None:
            _dtensor_singleton.clear_tpu_core_ids()
        _dtensor_singleton = None

@ops.RegisterGradient("Relayout")
def _relayout_gradient(op, grad):
    grad = gen_dtensor_ops.relayout_like(grad, layout_input=op.inputs[0])
    return grad

@ops.RegisterGradient("RelayoutLike")
def _relayout_grad_gradient(op, grad):
    grad = gen_dtensor_ops.relayout_like(grad, layout_input=op.inputs[0])
    return grad, None

@ops.RegisterGradient("CopyToMesh")
def _copy_to_mesh_gradient(op, grad):
    grad = gen_dtensor_ops.copy_to_mesh_grad(grad, forward_input=op.inputs[0])
    return grad

@ops.RegisterGradient("CopyToMeshGrad")
def _copy_to_mesh_grad_gradient(op, grad):
    grad = gen_dtensor_ops.copy_to_mesh_grad(grad, forward_input=op.inputs[0])
    return grad, None
