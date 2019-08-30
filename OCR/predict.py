import tensorflow as tf
import pipe_2 as pipe
from transformer import transformer_main
from transformer.model import model_params
import numpy as np
import pandas as pd
import subprocess
import pickle

tf.logging.set_verbosity(tf.logging.INFO)
PARAMS = model_params.BIG_MULTI_GPU_PARAMS
MODEL_NUM = 17
MODEL_DIR = 'transformer/model_directory_old/model_' + str(MODEL_NUM)

def predict_bunch(est, step):

    pred = est.predict(
        input_fn=lambda: pipe.csv_input_fn('test_parse', PARAMS['default_batch_size'])
    )
    pred = list(pred)
    pickle.dump(pred, open('predictions/pickled_predictions_' + str(step), 'wb'))


run_config = tf.estimator.RunConfig(
    model_dir=MODEL_DIR,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=1
)

estimator = tf.estimator.Estimator(
    model_fn=transformer_main.model_fn,
    config=run_config,
    params=PARAMS
)

df = pd.read_csv('data/test_dataset.csv', dtype=np.str)

chunk_size = 100

for i in range(200, df.shape[0], chunk_size):
    print(i)
    df.iloc[i:i + chunk_size].to_csv('data/test_parse_dataset.csv', index=False, encoding='UTF-8')
    predict_bunch(estimator, i)

#pickle.dump(predictions, open('pickled_predictions', 'wb'))
predictions = []
for i in range(0, 40000, 100):
    parse = pickle.load(open("predictions/pickled_predictions_"+str(i), "rb"))
    parse = [x["outputs"] for x in parse]
    predictions += parse

df.iloc[:100].to_csv('data/test_parse_dataset.csv', index=False, encoding='UTF-8')
predict_bunch(estimator, 0)