import tensorflow as tf
import pipe_2 as pipe
from transformer import transformer_main
from transformer.model import model_params
import subprocess

tf.logging.set_verbosity(tf.logging.INFO)
PARAMS = model_params.BASE_MULTI_GPU_PARAMS
MODEL_NUM = 16
MODEL_DIR = 'transformer/model_directory_old/model_' + str(MODEL_NUM)


run_config = tf.estimator.RunConfig(
    model_dir=MODEL_DIR,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=1
)

estimator = tf.estimator.Estimator(
    model_fn=tf.contrib.estimator.replicate_model_fn(transformer_main.model_fn),
    config=run_config,
    params=PARAMS
)

estimator.evaluate(
    input_fn=lambda: pipe.csv_input_fn('val', PARAMS['default_batch_size']),
    steps=100,
)


