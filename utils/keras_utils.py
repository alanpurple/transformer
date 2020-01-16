import multiprocessing
import os
import time

from absl import logging
import tensorflow as tf
from tensorflow_core.core.protobuf import rewriter_config_pb2
from tensorflow_core.python import tf2
from tensorflow_core.python.eager import profiler
from tensorflow_core.python.keras import callbacks

class BatchTimestamp(object):

    def __init__(self,batch_index,timestamp):
        self.batch_index=batch_index
        self.timestamp=timestamp

    def __repr__(self):
        return "'BatchTimestamp<batch_index: {}, timestamp {}'".format(self.batch_index,self.timestamp)

class TimeHistory(callbacks.Callback):

    def __init__(self,batch_size,log_steps):
        self.batch_size=batch_size
        super().__init__()
        self.log_steps=log_steps
        self.global_steps=0

        # Logs start of step 1 then end of each step based on log_steps interval.
        self.timestamp_log=[]

        # Records the time each epoch takes to run from start to finish of epoch.
        self.epoch_runtime_log=[]

    def on_train_end(self,logs=None):
        self.train_finish_time=time.time()
    
    def on_epoch_begin(self,epoch,logs=None):
        self.epoch_start=time.time()

    def on_batch_begin(self,batch,logs=None):
        self.global_steps+=1
        if self.global_steps==1:
            self.start_time = time.time()
            self.timestamp_log.append(BatchTimestamp(self.global_steps,self.start_time))

    def on_batch_end(self,batch,logs=None):
        if self.global_steps % self.log_steps == 0:
            timestamp=time.time()
            elapsed_time=timestamp-self.start_time
            examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
            self.timestamp_log.append(BatchTimestamp(self.global_steps, timestamp))
            logging.info(
                "BenchmarkMetric: {'global step':%d, 'time_taken': %f,"
                "'examples_per_second': %f}",
                self.global_steps, elapsed_time, examples_per_second)
            self.start_time = timestamp

    def on_epoch_end(self,epoch,logs=None):
        epoch_run_time=time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)
        logging.info(
            "BenchmarkMetric: {'epoch':%d, 'time_taken': %f}", epoch, epoch_run_time
        )

def get_profiler_callback(model_dir,profile_steps,enable_tensorboard,steps_per_epoch):
    """Validate profile_steps flag value and return profiler callback."""
    profile_steps_error_message=(
        'profile_steps must be a comma separated pair of positive integers, '
        'specifying the first and last steps to be profiled.'
    )
    try:
        profile_steps = [int(i) for i in profile_steps.split(',')]
    except ValueError:
        raise ValueError(profile_steps_error_message)
    if len(profile_steps) != 2:
        raise ValueError(profile_steps_error_message)
    start_step, stop_step = profile_steps
    if start_step < 0 or start_step > stop_step:
        raise ValueError(profile_steps_error_message)
    if enable_tensorboard:
        logging.warning(
            'Both TensorBoard and profiler callbacks are used. Note that the '
            'TensorBoard callback profiles the 2nd step (unless otherwise '
            'specified). Please make sure the steps profiled by the two callbacks '
            'do not overlap.')
    return ProfilerCallback(model_dir, start_step, stop_step, steps_per_epoch)

class ProfilerCallback(callbacks.Callback):
    """Save profiles in specified step range to log directory."""

    def __init__(self,log_dir,start_step,stop_step,steps_per_epoch):
        super().__init__()
        self.log_dir=log_dir
        self.start_step=start_step
        self.stop_step=stop_step
        self.start_epoch=start_step//steps_per_epoch
        self.stop_epoch=stop_step//steps_per_epoch
        self.start_step_in_epoch=start_step % steps_per_epoch
        self.stop_step_in_epoch=stop_step % steps_per_epoch
        self.should_start=False
        self.should_stop=False

    def on_epoch_begin(self,epoch,logs=None):
        if epoch==self.start_epoch:
            self.should_start=True
        if epoch==self.stop_epoch:
            self.should_stop=True

    def on_batch_begin(self,batch,logs=None):
        if batch == self.start_step_in_epoch and self.should_start:
            self.should_start = False
            profiler.start()
            logging.info('Profiler started at Step %s', self.start_step)

    def on_batch_end(self, batch, logs=None):
        if batch == self.stop_step_in_epoch and self.should_stop:
            self.should_stop = False
            results = profiler.stop()
            profiler.save(self.log_dir, results)
            logging.info(
                'Profiler saved profiles for steps between %s and %s to %s',
                self.start_step, self.stop_step, self.log_dir)

def set_session_config(enable_xla=False):
    if enable_xla:
        tf.config.optimizer.set_jit(True)

def set_gpu_thread_mode_and_count(gpu_thread_mode,
                                  datasets_num_private_threads,
                                  num_gpus, per_gpu_thread_count):
    """Set GPU thread mode and count, and adjust dataset threads count."""
    cpu_count = multiprocessing.cpu_count()
    logging.info('Logical CPU cores: %s', cpu_count)

    # Allocate private thread pool for each GPU to schedule and launch kernels
    per_gpu_thread_count = per_gpu_thread_count or 2
    os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    logging.info('TF_GPU_THREAD_COUNT: %s',
                os.environ['TF_GPU_THREAD_COUNT'])
    logging.info('TF_GPU_THREAD_MODE: %s',
                os.environ['TF_GPU_THREAD_MODE'])

    # Limit data preprocessing threadpool to CPU cores minus number of total GPU
    # private threads and memory copy threads.
    total_gpu_thread_count = per_gpu_thread_count * num_gpus
    num_runtime_threads = num_gpus
    if not datasets_num_private_threads:
        datasets_num_private_threads = min(
            cpu_count - total_gpu_thread_count - num_runtime_threads,
            num_gpus * 8)
        logging.info('Set datasets_num_private_threads to %s',
                        datasets_num_private_threads)