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
    b = convert_tokens(tokens.values[4])
    c = convert_tokens(tokens.values[5])
    d = convert_tokens(tokens.values[6])
    tok = {
        'line1': a,
        'line2': b,
        'line3': c,
        'line4': d
    }
    return image, tok


dat = tf.data.TextLineDataset('dataset/2017').skip(1)
dat = dat.map(preprocess, num_parallel_calls=1)

dat = dat.padded_batch(
        batch_size=3,
        padded_shapes=([80,300,1],{
                'line1': [-1],
                'line2': [-1],
                'line3': [-1],
                'line4': [-1],
            }),
        padding_values=(0.0,{
                'line1': 0,
                'line2': 0,
                'line3': 0,
                'line4': 0
            }),
        drop_remainder=True
    )
dat = dat.prefetch(1)

data,test = dat.make_one_shot_iterator().get_next()
print(data)
with tf.Session() as sess:
    print(sess.run(test['line1']))
    print(sess.run(test['line2']))
    print(sess.run(test['line3']))
    print(sess.run(test['line4']))
