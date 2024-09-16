"""
A heartbeat service periodically pinging all workers.

This module implements a heartbeat mechanism to detect worker failures or restarts
in a distributed system. Workers exchange a randomly generated number until normal
program termination. If any worker stops or restarts, other workers will detect
that and terminate themselves.

Copyright 2022 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
"""

import atexit
import threading
import time
from typing import Optional

import numpy as np
import tensorflow as tf

from tensorflow.dtensor.python import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.platform import tf_logging as logging

# Constants
CONSECUTIVE_FAILURES_LIMIT = 3
DEFAULT_TIMEOUT = 2

class HeartbeatService:
    def __init__(self):
        self._failure_count = 0
        self._heartbeat_timer: Optional[threading.Event] = None

    def _heartbeat(self, period: int, timer: threading.Event, token: int,
                   num_tasks: int, task_id: int, device: tf_device.DeviceSpec):
        """Periodically sends and receives a heartbeat signal."""
        logging.info('Starting a heartbeat thread')
        
        while not timer.wait(period):
            signal = np.zeros(num_tasks, dtype=np.int32)
            signal[task_id] = token

            logging.vlog(2, 'Sending heartbeat signal %s', signal)
            try:
                with tf.device(device):
                    signal = tf.raw_ops.AllReduce(
                        input=tf.constant(signal),
                        group_size=num_tasks,
                        group_key=0,
                        instance_key=0,
                        merge_op='Add',
                        final_op='Id',
                        timeout_seconds=max(period - 10, DEFAULT_TIMEOUT)
                    ).numpy()
            except Exception as e:
                self._handle_failure(e)
                continue

            logging.vlog(2, 'Received heartbeat signal %s', signal)
            if not np.all(signal == token):
                logging.fatal('Unexpected heartbeat signal received: %s', signal)

            self._failure_count = 0  # Reset on success

        logging.info('Exiting the heartbeat thread normally')

    def _handle_failure(self, exception: Exception):
        """Handle heartbeat failures."""
        self._failure_count += 1
        if self._failure_count < CONSECUTIVE_FAILURES_LIMIT:
            logging.warning('Heartbeat failure %d, %d more until limit: %s',
                            self._failure_count,
                            CONSECUTIVE_FAILURES_LIMIT - self._failure_count, exception)
        else:
            logging.fatal('Heartbeat failure %d, limit of %d reached: %s',
                          self._failure_count, CONSECUTIVE_FAILURES_LIMIT, exception)

    def start(self, period: int) -> threading.Event:
        """Starts a persistent thread exchanging heartbeats between workers."""
        if self._heartbeat_timer is not None:
            logging.warning('A heartbeat thread is already running, skipping this one.')
            return self._heartbeat_timer

        task_id = config.client_id()
        num_tasks = config.num_clients()

        token = self._initialize_token(task_id, num_tasks)
        device = self._get_device_spec(task_id)

        self._heartbeat_timer = threading.Event()
        atexit.register(self._stop_heartbeat, period)

        thread = threading.Thread(
            target=self._heartbeat,
            args=[period, self._heartbeat_timer, token, num_tasks, task_id, device],
            daemon=True
        )
        thread.start()

        return self._heartbeat_timer

    def _initialize_token(self, task_id: int, num_tasks: int) -> int:
        """Initialize the heartbeat token."""
        if task_id == 0:
            token = np.random.randint(0, 2**16 - 1)
            signal = np.full(num_tasks, token, dtype=np.int32)
        else:
            signal = np.zeros(num_tasks, dtype=np.int32)
        
        logging.info('Initial heartbeat signal: %s', signal)
        
        device = self._get_device_spec(task_id)
        with tf.device(device):
            signal = tf.raw_ops.AllReduce(
                input=tf.constant(signal),
                group_size=num_tasks,
                group_key=0,
                instance_key=0,
                merge_op='Add',
                final_op='Id',
                timeout_seconds=DEFAULT_TIMEOUT
            ).numpy()
        
        logging.info('Merged heartbeat signal %s', signal)
        
        if task_id == 0:
            if not np.all(signal == token):
                logging.fatal('Merged heartbeat signal has value != %d', token)
        else:
            if len(set(signal)) != 1:
                logging.fatal('Merged heartbeat signal has unequal elements')
            token = signal[0]
        
        return token

    @staticmethod
    def _get_device_spec(task_id: int) -> tf_device.DeviceSpec:
        """Get the device specification for the current task."""
        return tf_device.DeviceSpec(
            job=config.job_name(),
            replica=0,
            task=task_id,
            device_type='CPU',
            device_index=0
        )

    def _stop_heartbeat(self, period: int):
        """Stop the heartbeat thread."""
        logging.info('Stopping the heartbeat thread')
        if self._heartbeat_timer:
            self._heartbeat_timer.set()
            time.sleep(max(period // 10, DEFAULT_TIMEOUT))

# Global instance
heartbeat_service = HeartbeatService()

def start(period: int) -> threading.Event:
    """Start the heartbeat service."""
    return heartbeat_service.start(period)
