import collections
from typing import Dict, List, Union

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export

@tf_export('experimental.dtensor.sharded_save', v1=[])
def sharded_save(
    mesh: layout_lib.Mesh,
    file_prefix: Union[str, tensor_lib.Tensor],
    tensor_names: Union[List[str], tensor_lib.Tensor],
    shape_and_slices: Union[List[str], tensor_lib.Tensor],
    tensors: List[Union[tensor_lib.Tensor, tf_variables.Variable]],
):
    with ops.device(api.device_name()):
        io_ops.save_v2(file_prefix, tensor_names, shape_and_slices, tensors)

    mesh_util.barrier(mesh.host_mesh(), 'SaveV2')

    with api.default_mesh(mesh.host_mesh()):
        merge_op = io_ops.MergeV2Checkpoints(
            checkpoint_prefixes=[file_prefix],
            destination_prefix=file_prefix,
            delete_old_dirs=True)

    mesh_util.barrier(mesh.host_mesh(), 'MergeV2Checkpoints')

    return merge_op

@tf_export('experimental.dtensor.enable_save_as_bf16', v1=[])
def enable_save_as_bf16(variables: List[tf_variables.Variable]):
    for v in variables:
        if isinstance(v, d_variable.DVariable):
            v.save_as_bf16 = True

@tf_export('experimental.dtensor.name_based_restore', v1=[])
def name_based_restore(
    mesh: layout_lib.Mesh,
    checkpoint_prefix: str,
    name_tensor_dict: Dict[
        str, Union[tensor_lib.Tensor, tf_variables.Variable]],
):
    if not context.executing_eagerly():
        raise ValueError('name based restore must run eagerly.')

    ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)

    for name, tensor in ordered_name_tensor_dict.items():
        try:
            if api.fetch_layout(tensor).mesh.device_type().upper() != 'CPU':
                raise ValueError(
                    f'Restoring a non CPU Tensor is not supported currently. Offending '
                    f'tensor name : {name}')
        except errors_impl.OpError as op_error:
            raise ValueError(
                'Saving/Restoring tensor must be a DTensor') from op_error

    checkpoint_prefix = api.pack(
        [checkpoint_prefix] * mesh.num_local_devices(),
        layout_lib.Layout.replicated(mesh.host_mesh(), rank=0))
    tensor_names = api.pack(
        [list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(),
        layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    shape_and_slices = api.pack(
        [[''] * len(ordered_name_tensor_dict)] * mesh.num_local_devices(),
        layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    input_shapes = [tensor.shape for tensor in ordered_name_tensor_dict.values()]
    input_layouts = [
        api.fetch_layout(tensor).to_string()
        for tensor in ordered_name_tensor_dict.values()
    ]

    with ops.device(api.device_name()):
        restored_cpu_tensors = gen_dtensor_ops.d_tensor_restore_v2(
            prefix=checkpoint_prefix,
            tensor_names=tensor_names,
            shape_and_slices=shape_and_slices,
            input_shapes=input_shapes,
            input_layouts=input_layouts,
            dtypes=[tensor.dtype for tensor in ordered_name_tensor_dict.values()],
        )

    return collections.OrderedDict(
        zip(ordered_name_tensor_dict.keys(), restored_cpu_tensors)
    )

@tf_export('experimental.dtensor.name_based_save', v1=[])
def name_based_save(
    mesh: layout_lib.Mesh,
    checkpoint_prefix: Union[str, tensor_lib.Tensor],
    name_tensor_dict: Dict[
        str, Union[tensor_lib.Tensor, tf_variables.Variable]],
):
    if not context.executing_eagerly():
        raise ValueError('name based save must run eagerly.')

    ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)

    checkpoint_prefix = api.pack([checkpoint_prefix] * mesh.num_local_devices(),
                                 layout_lib.Layout.replicated(
                                     mesh.host_mesh(), rank=0))
    tensor_names = api.pack(
        [list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(),
        layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))

    sharded_save(
        mesh,
        file_prefix=checkpoint_prefix,
        tensor_names=tensor_names,
        shape_and_slices=[''] * len(ordered_name_tensor_dict),
        tensors=list(ordered_name_tensor_dict.values()))
