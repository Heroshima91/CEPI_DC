import tensorflow as tf


def convert_tokens(tokens):

    tokens = tf.strings.split([tokens], sep=' ', maxsplit=None)
    tokens = tf.cond(
        pred=tf.equal(tokens.values, [''])[0],
        true_fn=lambda: tf.constant(['1']),
        false_fn=lambda: tf.concat([tokens.values, ['1']], axis=0),
    )
    tokens = tf.strings.to_number(tokens,tf.int32)

    return tokens


def preprocess(file):
    tokens = tf.strings.split([file], sep=',', maxsplit=None)
    path = tokens.values[1]
    image = tf.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_images(image, [80, 300])
    image /= 255
    a = convert_tokens(tokens.values[3])
    return image, a


def input_fn(dataset_name,batch_size):
    dat = tf.data.TextLineDataset('dataset/'+dataset_name).skip(1)
    #dat = dat.apply(tf.data.experimental.shuffle_and_repeat(20000))
    dat = dat.map(preprocess, num_parallel_calls=batch_size)
    dat = dat.padded_batch(
        batch_size=batch_size,
        padded_shapes=([80, 300, 1],[-1]),
        padding_values=(0.0, 0),
        drop_remainder=True
    )
    dat = dat.prefetch(batch_size)

    # TODO put that back for GPU prefetching
    #dat = dat.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))
    return dat





"""

#TEST pipe.py
image, tar = input_fn("2017.csv",2000).make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print(sess.run(tar))"""





