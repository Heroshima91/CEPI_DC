import tensorflow as tf
import pipe
from transformer import transformer_main
from transformer.model import model_params
import os
import subprocess

tf.logging.set_verbosity(tf.logging.INFO)
PARAMS = model_params.TINY_PARAMS
MODEL_DIR = 'transformer/model_directory/'


def get_next_model_dir():
    list_name = [int(name[name.find('_') + 1:]) for name in os.listdir('transformer/model_directory')]

    if len(list_name) is 0:
        last_model = 0
    else:
        last_model = max(list_name)

    return MODEL_DIR + '/model_' + str(last_model + 1)


run_config = tf.estimator.RunConfig(
    model_dir=get_next_model_dir(),
    save_checkpoints_steps=5000,
    keep_checkpoint_max=1
)

estimator = tf.estimator.Estimator(
    model_fn=tf.contrib.estimator.replicate_model_fn(transformer_main.model_fn),
    config=run_config,
    params=PARAMS
)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: pipe.input_fn('train.csv', PARAMS['default_batch_size']),
    max_steps=500000
)

eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: pipe.input_fn('eval.csv', PARAMS['default_batch_size']),
    steps=100,
    start_delay_secs=60,
    throttle_secs=120
)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

subprocess.call('py -m tensorboard.main --logdir=C:/Users/louis/PycharmProjects/NLP_seq2seq_final/transformer/model_directory --host localhost --port 8088')


