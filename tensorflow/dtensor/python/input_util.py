"""
APIs to efficiently handle input datasets in DTensor.

This module provides the DTensorDataset class, which efficiently loads input data
and correctly packs it to corresponding devices for use with DTensor. It's designed
to work with unbatched data and supports both data and model parallel setups.

Copyright 2022 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
"""

import dataclasses
from typing import Any, List, Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.dtensor.python import api, config, layout as layout_lib
from tensorflow.python.data.experimental.ops import data_service_ops, distribute
from tensorflow.python.framework import constant_op, dtypes, tensor_shape, tensor_spec
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@dataclasses.dataclass
class TFDataServiceConfig:
    """Specifies the tf.data service configuration to use."""
    dispatcher_address: str
    job_name: str

class _DTensorIterator(tf.data.Iterator):
    """An iterator for a tf.data.Dataset distributed using DTensor."""

    def __init__(self, dtensor_components: Tuple[tf.Tensor],
                 global_element_spec: tf.TensorSpec, layouts: Any):
        self._iterator_resource_dtensor, = dtensor_components
        self._global_element_spec = global_element_spec
        self._layouts = layouts
        self._layouts_str = nest.map_structure(lambda layout: layout.to_string(), layouts)
        super().__init__(components=dtensor_components, element_spec=global_element_spec)

    def __next__(self):
        try:
            host_elem = self._next_internal()
            tf.compat.v1.get_default_session().run(tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS))
            device_elem = nest.map_structure(api.copy_to_mesh, host_elem, self._layouts)
            tf.compat.v1.get_default_session().run(tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS))
            return device_elem
        except tf.errors.OutOfRangeError as e:
            if tf.executing_eagerly():
                raise StopIteration from e
            else:
                raise e

    @property
    def _type_spec(self):
        return _DTensorIteratorSpec(self._global_element_spec, self._layouts_str)

class _DTensorIteratorSpec(tf.data.IteratorSpec):
    """Type specification for `_DTensorIterator`."""

    def __init__(self, global_element_spec: tf.TensorSpec, layouts_str: Any):
        super().__init__(global_element_spec)
        self._global_element_spec = global_element_spec
        self._layouts_str = layouts_str

    @property
    def value_type(self):
        return _DTensorIterator

    def _serialize(self):
        return (self._global_element_spec, self._layouts_str)

    @property
    def _component_specs(self):
        return (tf.TensorSpec([], dtypes.resource),)

    def _to_components(self, value):
        return (value._iterator_resource_dtensor,)

    def _from_components(self, components):
        layouts = nest.map_structure(layout_lib.Layout.from_string, self._layouts_str)
        return _DTensorIterator(components, self._global_element_spec, layouts)

    @classmethod
    def from_value(cls, value):
        return cls(value._global_element_spec, value._layouts_str)

def _validate_input(flattened_layouts: Sequence[layout_lib.Layout],
                    flattened_elem_spec: Sequence[tf.TensorSpec],
                    dataset_already_batched: bool):
    """Validates compatibility between dataset layouts and element specs."""
    if not flattened_elem_spec:
        raise ValueError('Expected input element spec of at least one element, was empty.')

    first_elem_shape = flattened_elem_spec[0].shape

    for layout, elem_spec in zip(flattened_layouts, flattened_elem_spec):
        if elem_spec.shape.rank is None:
            raise ValueError(f'Dataset element shape must have a valid rank, got spec {elem_spec}.')

        expected_rank = elem_spec.shape.rank + (0 if dataset_already_batched else 1)
        if layout.rank != expected_rank:
            raise ValueError(f'Expected layout with rank {expected_rank} for element spec {elem_spec}, '
                             f'got layout {layout.sharding_specs}. Check that the dataset is not '
                             'batched before passing to DTensorDataset.')

        if dataset_already_batched:
            batch_dim_size = first_elem_shape.as_list()[0]
            if batch_dim_size is None:
                raise ValueError(f'Size of batch dimension of element spec {elem_spec} is None. '
                                 'Ensure drop_remainder=True when batching the dataset.')
            if elem_spec.shape.as_list()[0] != batch_dim_size:
                raise ValueError(f'Size of batch dimension of element spec {elem_spec} does not '
                                 f'match expected size {batch_dim_size}.')

def _shard_counts(layout: layout_lib.Layout, batch_dim: Optional[str] = None) -> List[int]:
    """Computes the number of shards in each dimension of the layout."""
    return [1 if spec in (batch_dim, layout_lib.UNSHARDED) else layout.mesh.dim_size(spec)
            for spec in layout.sharding_specs]

def _index_matrix(layout: layout_lib.Layout, elem_spec: tf.TensorSpec) -> tf.Tensor:
    """Computes a utility matrix for deriving device-based slice offsets."""
    matrix = [[0 if spec in (layout_lib.UNSHARDED, dim) else
               elem_spec.shape[layout_idx] // layout.mesh.dim_size(dim)
               for layout_idx, spec in enumerate([None] + list(layout.sharding_specs[1:]))]
              for dim in layout.mesh.dim_names]
    return constant_op.constant(matrix, dtype=dtypes.int32)

def _pack_iterator_resource_dtensor(datasets: List[Tuple[int, tf.data.Dataset]],
                                    layouts: Any, mesh: layout_lib.Mesh,
                                    num_local_devices_per_replica: int) -> tf.Tensor:
    """Creates a DTensor iterator resource for the per-replica datasets."""
    host_mesh_devices = mesh.host_mesh().local_devices()
    iterators = []

    for _, dataset in datasets:
        for idx in range(num_local_devices_per_replica):
            with tf.device(host_mesh_devices[len(iterators)]):
                device_dataset = dataset.shard(num_shards=num_local_devices_per_replica, index=idx)
                iterators.append(iter(device_dataset))

    if len(iterators) != len(host_mesh_devices):
        raise ValueError(f'The `datasets` argument does not have the correct number of '
                         f'underlying datasets, found {len(iterators)} but expected '
                         f'{len(host_mesh_devices)}.')

    host_layouts = nest.map_structure(
        lambda l: layout_lib.Layout(l.sharding_specs, mesh.host_mesh()), layouts)

    iterator_resources = [it._iterator_resource for it in iterators]
    d_iterator_resource = api.pack(
        iterator_resources,
        layout_lib.Layout.replicated(mesh=mesh.host_mesh(), rank=0))
    api._dtensor_device().set_iterator_element_layouts(d_iterator_resource, nest.flatten(host_layouts))

    return d_iterator_resource

@tf_export('experimental.dtensor.DTensorDataset', v1=[])
class DTensorDataset(tf.data.Dataset):
    """A dataset of DTensors."""

    def __init__(self, dataset: tf.data.Dataset, *, mesh: layout_lib.Mesh, layouts: Any,
                 global_batch_size: int, dataset_already_batched: bool = False,
                 batch_dim: Optional[str] = None, prefetch: Optional[int] = None,
                 tf_data_service_config: Optional[TFDataServiceConfig] = None):
        """Creates a DTensorDataset."""
        super().__init__()
        self._init_args = locals()
        del self._init_args['self']
        del self._init_args['__class__']

        if tf_data_service_config is not None:
            raise NotImplementedError('Multi-client DTensorDataset is currently not supported. '
                                      'Check b/271162918.')

        self._setup_dataset(dataset, mesh, layouts, global_batch_size,
                            dataset_already_batched, batch_dim)
        self._setup_partitioning(mesh, batch_dim)
        self._prefetch = prefetch
        self._tf_data_service_config = tf_data_service_config

    def _setup_dataset(self, dataset, mesh, layouts, global_batch_size,
                       dataset_already_batched, batch_dim):
        """Sets up the dataset and validates inputs."""
        self._mesh = mesh
        self._layouts = layouts
        self._batch_dim = batch_dim

        nest.assert_same_structure(dataset.element_spec, layouts)
        flattened_layouts = nest.flatten(layouts)
        flattened_elem_spec = nest.flatten(dataset.element_spec)

        self._setup_replicas(batch_dim, mesh)
        _validate_input(flattened_layouts, flattened_elem_spec, dataset_already_batched)

        expected_batch_size = global_batch_size // self.num_global_replicas
        if not dataset_already_batched:
            self._batched_dataset = dataset.batch(expected_batch_size, drop_remainder=True)
        else:
            per_replica_batch_size = flattened_elem_spec[0].shape.as_list()[0]
            if per_replica_batch_size != expected_batch_size:
                raise ValueError(f'per_replica_batch_size does not match expected size based on '
                                 f'the mesh, got {per_replica_batch_size} but expected {expected_batch_size}.')
            self._batched_dataset = dataset

        self._setup_global_element_spec(global_batch_size)

    def _setup_replicas(self, batch_dim, mesh):
        """Sets up replica-related attributes."""
        if batch_dim:
            self.num_global_replicas = mesh.dim_size(batch_dim)
            self._local_replica_ids = list(dict.fromkeys(
                [loc[batch_dim] for loc in mesh.local_device_locations()]))
            for layout in nest.flatten(self._layouts):
                if batch_dim != layout.sharding_specs[0]:
                    raise ValueError(f'batch_dim {batch_dim} was specified but at least one '
                                     f'layout did not contain it: {layout}')
        else:
            self.num_global_replicas = 1
            self._local_replica_ids = [0]

    def _setup_global_element_spec(self, global_batch_size):
        """Sets up the global element spec for the dataset."""
        flattened_global_elem_spec = []
        batch_tensor_shape = tensor_shape.as_shape([global_batch_size])
        for elem_spec in nest.flatten(self._batched_dataset.element_spec):
            new_elem_spec = tensor_spec.TensorSpec(
                shape=tf.TensorShape(batch_tensor_shape).concatenate(elem_spec.shape[1:]),
                dtype=elem_spec.dtype,
                name=elem_spec.name)
            flattened_global_elem_spec.append(new_elem_spec)
        self._global_element_spec = nest.pack_sequence_as(
            self._batched_dataset.element_spec, flattened_global_elem_spec)

    def _setup_partitioning(self, mesh, batch_dim):
        """Sets up partitioning-related attributes."""
        num_global_devices_per_replica = config.num_global_devices(
            mesh.device_type()) // self.num_global_replicas
        self._num_local_replicas = len(self._local_replica_ids)
        self._num_local_devices_per_replica = mesh.num_local_devices() // self._num_local_replicas
        self._num_clients_per_replica = (
            num_global_devices_per_replica // self._num_local_devices_per_replica)
        self._partition_offset = (config.client_id() % self._num_clients_per_replica
                                 ) * self._num_local_devices_per_replica

        flattened_layouts = nest.flatten(self._layouts)
        flattened_elem_spec = nest.flatten(self._batched_dataset.element_spec)
        self._all_shard_counts = [_shard_counts(layout, batch_dim) for layout in flattened_layouts]
        self._index_matrices = [_index_matrix(layout, elem_spec)
                                for layout, elem_spec in zip(flattened_layouts, flattened_elem_spec)]

    def __iter__(self):
        datasets = self._prepare_datasets()
        d_iterator_resource = _pack_iterator_resource_dtensor(
            datasets=datasets,
            layouts=self._layouts,
            mesh=self._mesh,
            num_local_devices_per_replica=self._num_local_devices_per_replica)

        return _DTensorIterator(
            dtensor_components=(d_iterator_resource,),
            global_element_spec=self._global_element_spec,
            layouts=self._layouts)

    def _prepare_datasets(self):
        """Prepares the datasets for each replica."""
        datasets = []
        local_dataset = self._batched_dataset

        sharding_policy = self._get_sharding_policy()
        local_dataset = self._apply_distribution(local_dataset, sharding_policy)

        for local_replica_idx, replica_id in enumerate(self._local_replica_ids):
            dataset = self._prepare_replica_dataset(local_dataset, local_replica_idx)
            datasets.append((replica_id, dataset))

        return datasets

    def _get_sharding_policy(self):
        """Determines the sharding policy based on the batch dimension and client setup."""
        if self._batch_dim is not None:
            if self._num_clients_per_replica > 1:
                return data_service_ops.ShardingPolicy.DATA
            else:
                return data_service_ops.ShardingPolicy.FILE
        else:
            return data_service_ops.ShardingPolicy.OFF

    def _apply_distribution(self, dataset, sharding_policy):
        """Applies distribution to the dataset if tf_data_service_config is specified."""
        if self._tf_data_service_config is not None:
            return dataset.apply(
                data_service_ops.distribute(
                    processing_mode=sharding_policy,
                    service=self._tf_data_service_config.dispatcher_address,
                    job_name=f'{self._tf_data_service_config.job_name}_{config.client_id()}',
                    target_workers='LOCAL'))
        return dataset

def _prepare_replica_dataset(self, dataset, local_replica_idx):
        """Prepares the dataset for a single replica."""
        dataset = distribute._AutoShardDataset(
            dataset,
            num_workers=self._num_local_replicas,
            index=local_replica_idx,
            num_replicas=self.num_global_replicas)

        dataset = self._repeat_batch(dataset, self._num_local_devices_per_replica)
        dataset = self._partition(dataset)

        if self._prefetch is not None:
            dataset = dataset.prefetch(self._prefetch * self._num_local_devices_per_replica)

        return dataset

    def _repeat_batch(self, dataset, repeats):
        """Repeats each batch in the dataset."""
        if repeats == 1:
            return dataset

        def repeat(*x):
            return tf.data.Dataset.from_tensors(x).repeat(repeats)

        return dataset.flat_map(repeat)

    def _partition(self, dataset):
        """Slices each dataset element on any sharded non-batch dimension."""
        if self._num_local_devices_per_replica == 1 and self._partition_offset == 0:
            return dataset

        def slice_batch(index, batch):
            flattened_batch = tf.nest.flatten(batch)
            flattened_output = []

            norm_index = tf.cast(index % self._num_local_devices_per_replica, dtype=tf.int32)
            norm_index += self._partition_offset
            coords = self._mesh.coords(norm_index)
            coords = tf.reshape(coords, (1, -1))

            for element, shard_counts, idx_matrix in zip(flattened_batch,
                                                         self._all_shard_counts,
                                                         self._index_matrices):
                indexes = tf.matmul(coords, idx_matrix)
                start = tf.reshape(indexes, (-1,))
                size = tf.shape(element, out_type=tf.int32) // shard_counts
                flattened_output.append(tf.slice(element, begin=start, size=size))

            return tf.nest.pack_sequence_as(batch, flattened_output)

        enumerated_dataset = dataset.enumerate()
        partitioned_dataset = enumerated_dataset.map(slice_batch)
        return partitioned_dataset

    @property
    def element_spec(self):
        return self._global_element_spec

    def _inputs(self):
        return []

    @property
    def _variant_tensor(self):
        return self._batched_dataset._variant_tensor

    def _to_variant_tensor(self):
        return self._batched_dataset._to_variant_tensor()

    def _as_variant_tensor(self):
        return self._batched_dataset._as_variant_tensor()

    def _as_serialized_graph(self):
        return self._batched_dataset._as_serialized_graph()

    def __getattr__(self, name):
        """Fallback to the original dataset for unsupported methods."""
        if hasattr(self._batched_dataset, name):
            return getattr(self._batched_dataset, name)
        raise AttributeError(f"'DTensorDataset' object has no attribute '{name}'")

    def __getitem__(self, index):
        """Implement indexing for DTensorDataset."""
        if isinstance(index, slice):
            return self._batched_dataset[index]
        raise TypeError("DTensorDataset indexing is not supported.")

    def __len__(self):
        """Return the number of elements in the dataset."""
        return len(self._batched_dataset)

    def __repr__(self):
        return f"DTensorDataset(mesh={self._mesh}, layouts={self._layouts})"

# Helper functions that could be useful for testing or debugging

def create_test_dataset(num_elements: int = 100) -> tf.data.Dataset:
    """Creates a simple dataset for testing purposes."""
    return tf.data.Dataset.range(num_elements)

def create_test_mesh(mesh_shape: List[int], device_type: str = 'CPU') -> layout_lib.Mesh:
    """Creates a test mesh with the given shape."""
    mesh_dims = [f'dim_{i}' for i in range(len(mesh_shape))]
    return layout_lib.Mesh(mesh_dims, mesh_shape, device_type)

def create_test_layout(mesh: layout_lib.Mesh, sharding_specs: List[str]) -> layout_lib.Layout:
    """Creates a test layout with the given sharding specs."""
    return layout_lib.Layout(sharding_specs, mesh)

# Example usage
if __name__ == "__main__":
    # Create a test dataset
    dataset = create_test_dataset(1000)

    # Create a test mesh
    mesh = create_test_mesh([2, 2])

    # Create a test layout
    layout = create_test_layout(mesh, ['dim_0', 'unsharded'])

    # Create a DTensorDataset
    dtensor_dataset = DTensorDataset(
        dataset=dataset,
        mesh=mesh,
        layouts=layout,
        global_batch_size=32,
        batch_dim='dim_0'
    )

    # Iterate over the dataset
    for batch in dtensor_dataset:
        print(batch)
        break  # Just print the first batch for this example
