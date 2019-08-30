import numpy as np 
import tensorflow as tf


data_test = np.loadtxt('dataset/test_data.csv', delimiter=' ', dtype='float32')
lab_test  = np.genfromtxt('dataset/test_label.csv', dtype='unicode', delimiter='\n')

run_config = tf.estimator.RunConfig(
    model_dir="model_dir/model27/",
    save_checkpoints_steps=2500,
    keep_checkpoint_max=1,
)

estimator = tf.estimator.Estimator(
    model_fn=model.model_fn,
    config=run_config,
    params=hparams
)

prediction = list(estimator.predict(
    input_fn=data_test
))

a = [arr.tolist() for arr in prediction]
b = []
for elt in a:
    c = elt[:4]
    b.append(c)
b = np.array(b)
lab_list = lab_test.tolist()
lab_int = []
for elt in lab_list:
    lab_int.append([int(d) for d in str(elt)])
lab_int = np.array(lab_int)
sec = np.sum(np.all(b == lab_int, axis=1))
number_of_equal_elements = np.sum(b==lab_int)
total_elements = np.multiply(*b.shape)
percentage = number_of_equal_elements/total_elements

print('total number of elements: \t\t{}'.format(total_elements))
print('number of identical elements: \t\t{}'.format(number_of_equal_elements))
print('number of different elements: \t\t{}'.format(total_elements-number_of_equal_elements))
print('percentage of identical elements: \t{:.2f}%'.format(percentage*100))
