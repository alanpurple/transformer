from absl import app,flags,logging
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import compute_bleu,transformer,optimizer,translate,data_pipeline
from utils import metrics,keras_utils,tokenizer,misc
from utils.flags import core as flags_core
import os

INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1

def translate_and_compute_bleu(model,
                               params,
                               subtokenizer,
                               bleu_source,
                               bleu_ref):
  """Translate file and report the cased and uncased bleu scores.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  # Create temporary file to store translation.
  output = 'out'

  translate.translate_file(
      model,
      params,
      subtokenizer,
      bleu_source,
      output_file=output,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, output, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, output, True)
  os.remove(output)
  return uncased_score, cased_score

def evaluate_and_log_bleu(model,
                          params,
                          bleu_source,
                          bleu_ref,
                          vocab_file,
                          distribution_strategy=None):
  """Calculate and record the BLEU score.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    vocab_file: A file containing the vocabulary for translation.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, params, subtokenizer, bleu_source, bleu_ref)

  logging.info("Bleu score (uncased): %s", uncased_score)
  logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score

class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.

    Raises:
      ValueError: if not using static batch for input data on TPU.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["num_gpus"] = num_gpus
    params["use_ctl"] = flags_obj.use_ctl
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_length"] = flags_obj.max_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_decode"] = flags_obj.padded_decode
    params["num_parallel_calls"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_tensorboard"] = flags_obj.enable_tensorboard
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
    params["steps_between_evals"] = flags_obj.steps_between_evals

    logging.info("Running transformer with num_gpus = %d", num_gpus)

    if params["dtype"] == tf.float16:
      # TODO(reedwm): It's pretty ugly to set the global policy in a constructor
      # like this. What if multiple instances of TransformerTask are created?
      # We should have a better way in the tf.keras.mixed_precision API of doing
      # this.
      loss_scale = flags_core.get_loss_scale(
          flags_obj, default_for_fp16="dynamic")
      policy = mixed_precision.Policy("mixed_float16", loss_scale=loss_scale)
      mixed_precision.set_policy(policy)

    elif params["dtype"] == tf.bfloat16:
      policy = mixed_precision.Policy(
          "mixed_bfloat16")
      mixed_precision.set_policy(policy)

  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj
    # Sets config options.
    keras_utils.set_session_config(enable_xla=flags_obj.enable_xla)

    _ensure_dir(flags_obj.model_dir)
    model = transformer.create_model(params, is_train=True)
    opt = self._create_optimizer()

    current_step = 0
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        logging.info("Loaded checkpoint %s", latest_checkpoint)
        current_step = opt.iterations.numpy()

    if params["use_ctl"]:
        train_loss_metric = tf.keras.metrics.Mean(
            "training_loss", dtype=tf.float32)
        if params["enable_tensorboard"]:
            summary_writer = tf.compat.v2.summary.create_file_writer(
                flags_obj.model_dir)
        else:
            summary_writer = tf.compat.v2.summary.create_noop_writer()
        train_metrics = [train_loss_metric]
        if params["enable_metrics_in_training"]:
            train_metrics = train_metrics + model.metrics
    else:
        model.compile(opt)

    model.summary()

    train_ds = data_pipeline.train_input_fn(params)
    train_ds = train_ds.map(
        lambda x,y:((x, y),), num_parallel_calls=params["num_parallel_calls"])
    if params["use_ctl"]:
      train_ds_iterator = iter(train_ds)

    callbacks = self._create_callbacks(flags_obj.model_dir, 0, params)

    # TODO(b/139418525): Refactor the custom training loop logic.
    @tf.function
    def train_steps(iterator, steps):
      """Training steps function for TPU runs.

      Args:
        iterator: The input iterator of the training dataset.
        steps: An integer, the number of training steps.

      Returns:
        A float, the loss value.
      """

      def _step_fn(inputs):
        """Per-replica step function."""
        inputs, targets = inputs
        with tf.GradientTape() as tape:
          logits = model([inputs, targets], training=True)
          loss = metrics.transformer_loss(logits, targets,
                                          params["label_smoothing"],
                                          params["vocab_size"])

        # De-dupes variables due to keras tracking issues.
        tvars = list({id(v): v for v in model.trainable_variables}.values())
        grads = tape.gradient(loss, tvars)
        opt.apply_gradients(zip(grads, tvars))
        # For reporting, the metric takes the mean of losses.
        train_loss_metric.update_state(loss)

      for _ in tf.range(steps):
        train_loss_metric.reset_states()
        _step_fn(next(iterator))

    cased_score, uncased_score = None, None
    cased_score_history, uncased_score_history = [], []
    while current_step < flags_obj.train_steps:
      remaining_steps = flags_obj.train_steps - current_step
      train_steps_per_eval = (
          remaining_steps if remaining_steps < flags_obj.steps_between_evals
          else flags_obj.steps_between_evals)
      current_iteration = current_step // flags_obj.steps_between_evals

      logging.info(
          "Start train iteration at global step:{}".format(current_step))
      history = None
      if params["use_ctl"]:
        # Runs training steps.
        with summary_writer.as_default():
          train_steps(
              train_ds_iterator,
              tf.convert_to_tensor(train_steps_per_eval, dtype=tf.int32))
          current_step += train_steps_per_eval
          train_loss = train_loss_metric.result().numpy().astype(float)
          logging.info("Train Step: %d/%d / loss = %s", current_step,
                       flags_obj.train_steps, train_loss)

          if params["enable_tensorboard"]:
            for metric_obj in train_metrics:
              tf.summary.scalar(metric_obj.name, metric_obj.result(),
                                          current_step)

        checkpoint_name = checkpoint.save(
            os.path.join(flags_obj.model_dir,
                         "ctl_step_{}.ckpt".format(current_step)))
        logging.info("Saved checkpoint to %s", checkpoint_name)
      else:
        history = model.fit(
            train_ds,
            initial_epoch=current_iteration,
            epochs=current_iteration + 1,
            steps_per_epoch=train_steps_per_eval,
            callbacks=callbacks,
            # If TimeHistory is enabled, progress bar would be messy. Increase
            # the verbose level to get rid of it.
            verbose=(2 if flags_obj.enable_time_history else 1))
        current_step += train_steps_per_eval
        logging.info("Train history: {}".format(history.history))

      logging.info("End train iteration at global step:{}".format(current_step))

      if (flags_obj.bleu_source and flags_obj.bleu_ref):
        uncased_score, cased_score = self.eval()
        cased_score_history.append([current_iteration + 1, cased_score])
        uncased_score_history.append([current_iteration + 1, uncased_score])

    stats = ({
        "loss": train_loss
    } if history is None else misc.build_stats(history, callbacks))
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
      stats["bleu_uncased_history"] = uncased_score_history
      stats["bleu_cased_history"] = cased_score_history
    return stats

  def eval(self):
    """Evaluates the model."""

    # We only want to create the model under DS scope for TPU case.
    # When 'distribution_strategy' is None, a no-op DummyContextManager will
    # be used.
    if not self.predict_model:
        self.predict_model = transformer.create_model(self.params, False)
    self._load_weights_if_possible(
        self.predict_model,
        tf.train.latest_checkpoint(self.flags_obj.model_dir))
    self.predict_model.summary()
    return evaluate_and_log_bleu(
        self.predict_model, self.params, self.flags_obj.bleu_source,
        self.flags_obj.bleu_ref, self.flags_obj.vocab_file)

  def predict(self):
    """Predicts result from the model."""
    params = self.params
    flags_obj = self.flags_obj

    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train=False)
      self._load_weights_if_possible(
          model, tf.train.latest_checkpoint(self.flags_obj.model_dir))
      model.summary()
    subtokenizer = tokenizer.Subtokenizer(flags_obj.vocab_file)

    ds = data_pipeline.eval_input_fn(params)
    ds = ds.map(lambda x, y: x).take(_SINGLE_SAMPLE)
    ret = model.predict(ds)
    val_outputs, _ = ret
    length = len(val_outputs)
    for i in range(length):
      translate.translate_from_input(val_outputs[i], subtokenizer)

  def _create_callbacks(self, cur_log_dir, init_steps, params):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)
    callbacks = misc.get_callbacks(params["steps_between_evals"])
    callbacks.append(scheduler_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_full_path, save_weights_only=True))
    return callbacks

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      logging.info("Load weights: {}".format(init_weight_path))
      # TODO(b/139414977): Having the same variable restoring method for both
      model.load_weights(init_weight_path)
    else:
      logging.info("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    # TODO(b/139414679): Explore the difference between using
    opt = tf.keras.optimizers.Adam(
        params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    if params["dtype"] == tf.float16:
      opt = mixed_precision.LossScaleOptimizer(
          opt,
          loss_scale=flags_core.get_loss_scale(
              self.flags_obj, default_for_fp16="dynamic"))
    if self.flags_obj.fp16_implementation == "graph_rewrite":
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    return opt

def _ensure_dir(log_dir):
  """Makes log dir if not existed."""
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)