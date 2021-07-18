import warnings

from keras.engine.training_utils import is_sequence, iter_sequence_infinite, should_run_validation
from keras.models import Model
import keras.callbacks as cbks
from keras.utils import OrderedEnqueuer, GeneratorEnqueuer
import keras.backend as K
from keras.utils.generic_utils import to_list


class ClassificationModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_generator(self, generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      validation_freq=1,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0):
        """Trains the model on data generated batch-by-batch by a Python generator
        (or an instance of `Sequence`).

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        The use of `keras.utils.Sequence` guarantees the ordering
        and guarantees the single use of every input per epoch when
        using `use_multiprocessing=True`.

        # Arguments
            generator: A generator or an instance of `Sequence`
                (`keras.utils.Sequence`) object in order to avoid
                duplicate data when using multiprocessing.
                The output of the generator must be either
                - a tuple `(inputs, targets)`
                - a tuple `(inputs, targets, sample_weights)`.
                This tuple (a single output of the generator) makes a single
                batch. Therefore, all arrays in this tuple must have the same
                length (equal to the size of this batch). Different batches may
                have different sizes. For example, the last batch of the epoch
                is commonly smaller than the others, if the size of the dataset
                is not divisible by the batch size.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Integer.
                Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch. It should typically
                be equal to `ceil(num_samples / batch_size)`
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire data provided,
                as defined by `steps_per_epoch`.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_data: This can be either
                - a generator or a `Sequence` object for the validation data
                - tuple `(x_val, y_val)`
                - tuple `(x_val, y_val, val_sample_weights)`
                on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
            validation_steps: Only relevant if `validation_data`
                is a generator. Total number of steps (batches of samples)
                to yield from `validation_data` generator before stopping
                at the end of every epoch. It should typically
                be equal to the number of samples of your
                validation dataset divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(validation_data)` as a number of steps.
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections.Container` instance (e.g. list, tuple, etc.). If an
                integer, specifies how many training epochs to run before a new
                validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only). This can be useful to tell the model to
                "pay more attention" to samples
                from an under-represented class.
            max_queue_size: Integer. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Maximum number of processes to spin up
                when using process-based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: Boolean.
                If `True`, use process-based threading.
                If unspecified, `use_multiprocessing` will default to `False`.
                Note that because this implementation
                relies on multiprocessing,
                you should not pass non-picklable arguments to the generator
                as they can't be passed easily to children processes.
            shuffle: Boolean. Whether to shuffle the order of the batches at
                the beginning of each epoch. Only used with instances
                of `Sequence` (`keras.utils.Sequence`).
                Has no effect when `steps_per_epoch` is not `None`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            ValueError: In case the generator yields data in an invalid format.

        # Example

        ```python
        def generate_arrays_from_file(path):
            while True:
                with open(path) as f:
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
        ```
        """

        """See docstring for `self.fit_generator`."""
        epoch = initial_epoch

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        use_sequence_api = is_sequence(generator)
        if not use_sequence_api and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the `keras.utils.Sequence'
                            ' class.'))
        if steps_per_epoch is None:
            if use_sequence_api:
                steps_per_epoch = len(generator)
            else:
                raise ValueError('`steps_per_epoch=None` is only valid for a'
                                 ' generator based on the '
                                 '`keras.utils.Sequence`'
                                 ' class. Please specify `steps_per_epoch` '
                                 'or use the `keras.utils.Sequence` class.')

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_use_sequence_api = is_sequence(validation_data)
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__') or
                   val_use_sequence_api)
        if (val_gen and not val_use_sequence_api and
                not validation_steps):
            raise ValueError('`validation_steps=None` is only valid for a'
                             ' generator based on the `keras.utils.Sequence`'
                             ' class. Please specify `validation_steps` or use'
                             ' the `keras.utils.Sequence` class.')

        # Prepare display labels.
        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        _callbacks = [cbks.BaseLogger(
            stateful_metrics=self.stateful_metric_names)]
        if verbose:
            _callbacks.append(
                cbks.ProgbarLogger(
                    count_mode='steps',
                    stateful_metrics=self.stateful_metric_names))
        _callbacks += (callbacks or []) + [self.history]
        callbacks = cbks.CallbackList(_callbacks)

        # it's possible to callback a different model than self:
        callback_model = self._get_callback_model()

        callbacks.set_model(callback_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks._call_begin_hook('train')

        enqueuer = None
        val_enqueuer = None

        try:
            if do_validation:
                if val_gen and workers > 0:
                    # Create an Enqueuer that can be reused
                    val_data = validation_data
                    if is_sequence(val_data):
                        val_enqueuer = OrderedEnqueuer(
                            val_data,
                            use_multiprocessing=use_multiprocessing)
                        validation_steps = validation_steps or len(val_data)
                    else:
                        val_enqueuer = GeneratorEnqueuer(
                            val_data,
                            use_multiprocessing=use_multiprocessing)
                    val_enqueuer.start(workers=workers,
                                       max_queue_size=max_queue_size)
                    val_enqueuer_gen = val_enqueuer.get()
                elif val_gen:
                    val_data = validation_data
                    if is_sequence(val_data):
                        val_enqueuer_gen = iter_sequence_infinite(val_data)
                        validation_steps = validation_steps or len(val_data)
                    else:
                        val_enqueuer_gen = val_data
                else:
                    # Prepare data for validation
                    if len(validation_data) == 2:
                        val_x, val_y = validation_data
                        val_sample_weight = None
                    elif len(validation_data) == 3:
                        val_x, val_y, val_sample_weight = validation_data
                    else:
                        raise ValueError('`validation_data` should be a tuple '
                                         '`(val_x, val_y, val_sample_weight)` '
                                         'or `(val_x, val_y)`. Found: ' +
                                         str(validation_data))
                    val_x, val_y, val_sample_weights = self._standardize_user_data(
                        val_x, val_y, val_sample_weight)
                    val_data = val_x + val_y + val_sample_weights
                    if self.uses_learning_phase and not isinstance(K.learning_phase(),
                                                                   int):
                        val_data += [0.]
                    for cbk in callbacks:
                        cbk.validation_data = val_data

            if workers > 0:
                if use_sequence_api:
                    enqueuer = OrderedEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing,
                        shuffle=shuffle)
                else:
                    enqueuer = GeneratorEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enqueuer.get()
            else:
                if use_sequence_api:
                    output_generator = iter_sequence_infinite(generator)
                else:
                    output_generator = generator

            callbacks.model.stop_training = False
            # Construct epoch logs.
            epoch_logs = {}
            while epoch < epochs:
                for m in self.stateful_metric_functions:
                    m.reset_states()
                steps_done = 0
                batch_index = 0
                steps_per_epoch = len(generator)

                callbacks.set_params({
                    'epochs': epochs,
                    'steps': steps_per_epoch,
                    'verbose': verbose,
                    'do_validation': do_validation,
                    'metrics': callback_metrics,
                })
                callbacks.on_epoch_begin(epoch)
                while steps_done < steps_per_epoch:
                    generator_output = next(output_generator)

                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    if x is None or len(x) == 0:
                        # Handle data tensors support when no input given
                        # step-size = 1 for data tensors
                        batch_size = 1
                    elif isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    # build batch logs
                    batch_logs = {'batch': batch_index, 'size': batch_size}
                    callbacks.on_batch_begin(batch_index, batch_logs)

                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)

                    outs = to_list(outs)
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)

                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    if (steps_done >= steps_per_epoch and
                            do_validation and
                            should_run_validation(validation_freq, epoch)):
                        # Note that `callbacks` here is an instance of
                        # `keras.callbacks.CallbackList`
                        if val_gen:
                            val_outs = self.evaluate_generator(
                                val_enqueuer_gen,
                                validation_steps,
                                callbacks=callbacks,
                                workers=0)
                        else:
                            # No need for try/except because
                            # data has already been validated.
                            val_outs = self.evaluate(
                                val_x, val_y,
                                batch_size=batch_size,
                                sample_weight=val_sample_weights,
                                callbacks=callbacks,
                                verbose=0)
                        val_outs = to_list(val_outs)
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

                    if callbacks.model.stop_training:
                        break

                callbacks.on_epoch_end(epoch, epoch_logs)
                epoch += 1
                if callbacks.model.stop_training:
                    break

        finally:
            try:
                if enqueuer is not None:
                    enqueuer.stop()
            finally:
                if val_enqueuer is not None:
                    val_enqueuer.stop()

        callbacks._call_end_hook('train')
        return self.history
