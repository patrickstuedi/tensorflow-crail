import tensorflow as tf
import tensorflow_crail as crail
dataset = crail.CrailDataset("/tmp.dat")
iterator = dataset.make_one_shot_iterator()
next_obj = iterator.get_next()
with tf.Session() as sess:
    for _ in range(3):
        print(sess.run(next_obj))
