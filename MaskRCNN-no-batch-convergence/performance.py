import tensorpack
import time

def humanize_float(num):
    return "{0:,.2f}".format(num)

class ThroughputTracker(tensorpack.Callback):
    """
    Calculate and display throughput of model, by keeping track of the duration of each step and each epoch. Saves and
    outputs throughput as items/second. Prints output and saves
    Args:
        items_per_step:         The number of items processed in each step
        items_per_epoch:        The number of items processed in each epoch
        trigger_every_n_steps:  If this argument is None, throughput will be calculated once per epoch. If this argument
                                is a number N, throughput will also be calculated and output every N steps. The step
                                counter starts over each epoch.
        log_fn:                 The function to call to display throughput in logs. If None, throughput
                                will not be printed. This argument does not impact saving throughput as tf.scalar
    """

    def __init__(self, items_per_step, items_per_epoch, trigger_every_n_steps=None, log_fn=None):
        self._items_per_step = items_per_step
        self._items_per_epoch = items_per_epoch
        self._trigger_every_n_steps = trigger_every_n_steps

        if log_fn is None:
            self._log_fn = lambda x: None    # Do nothing logger
        else:
            self._log_fn = log_fn

        self._step_counter = 0

    def _before_epoch(self):
        epoch_start_time = time.time()
        self._epoch_start_time = epoch_start_time
        self._step_start_time = epoch_start_time
        self._epoch_step_durations = []
        self._step_counter = 0



    def _trigger_step(self):
        self._step_end_time = time.time()
        step_duration = self._step_end_time - self._step_start_time
        self._epoch_step_durations.append(step_duration)

        if self._trigger_every_n_steps is not None:
            self._step_counter += 1
            if self._step_counter % self._trigger_every_n_steps == 0:
                sum_step_durations = sum(self._epoch_step_durations[-self._trigger_every_n_steps:]) / self._trigger_every_n_steps
                mean_step_duration = sum_step_durations

                log_prefix = f'[ThroughputTracker] Over last {self._trigger_every_n_steps} steps'
                self._log_fn(f'{log_prefix}, MeanDuration={humanize_float(mean_step_duration)} seconds')
                self._log_fn(f'{log_prefix}, MeanThroughput={humanize_float(self._items_per_step / mean_step_duration)} items/sec')
                self._step_counter = 0

        self._step_start_time = self._step_end_time

    def _after_epoch(self):
        self._epoch_end_time = time.time()

    def _trigger_epoch(self):
        epoch_run_clock_time = sum(self._epoch_step_durations)
        epoch_wall_clock_time = self._epoch_end_time - self._epoch_start_time

        overhead_time = epoch_wall_clock_time - epoch_run_clock_time
        mean_epoch_throughput = self._items_per_epoch / epoch_wall_clock_time

        log_prefix = "[ThroughputTracker] Over last epoch"
        self._log_fn(f'{log_prefix}, MeanEpochThroughput: {humanize_float(mean_epoch_throughput)}')
        self._log_fn(f'{log_prefix}, EpochWallClockDuration: {humanize_float(epoch_wall_clock_time)}')
        self._log_fn(f'{log_prefix}, CallbackOverheadDuration: {humanize_float(overhead_time)}')

        self.trainer.monitors.put_scalar("Throughput/MeanEpochThroughput", mean_epoch_throughput)
        self.trainer.monitors.put_scalar('Throughput/EpochWallClockDuration', epoch_wall_clock_time)
        self.trainer.monitors.put_scalar('Throughput/CallbackOverheadDuration', overhead_time)