from unittest import TestCase
import numpy as np
import tfg
import tensorflow as tf


class Test(TestCase):
    def test_evaluate(self):
        np.random.seed(10)
        inputs = np.random.uniform(size=(4, 64, 3))
        print("inputs ", inputs.min(), inputs.max())
        np.random.seed(20)
        outputs = np.random.uniform(size=(4, 64, 3))
        print("outputs ", outputs.min(), outputs.max())
        res = tfg.evaluate(inputs, outputs)
        with tf.Session() as sess:
            print(sess.run(res))

            inp_t = tf.convert_to_tensor(value=inputs)
            outp_t = tf.convert_to_tensor(value=outputs)
            inp_t_c = tf.expand_dims(inp_t, axis=-2)
            outp_t_c = tf.expand_dims(outp_t, axis=-3)
            diff = (inp_t_c - outp_t_c)
            res = sess.run([inp_t, outp_t, inp_t_c, outp_t_c, diff])
            print("ok")

            sq_dist = tf.reduce_sum(tf.multiply(diff, diff), axis=-1)

            min_sq_dist_a_to_b = tf.reduce_min(input_tensor=sq_dist, axis=-1)
            min_sq_dist_b_to_a = tf.reduce_min(input_tensor=sq_dist, axis=-2)
            chamf_dist = (
                    tf.reduce_mean(input_tensor=min_sq_dist_a_to_b, axis=-1) +
                    tf.reduce_mean(input_tensor=min_sq_dist_b_to_a, axis=-1))
            final_res = tf.reduce_mean(chamf_dist)
            res = sess.run([sq_dist, min_sq_dist_a_to_b, min_sq_dist_b_to_a, chamf_dist])
            print("ok")
            print(res[-1] / 2)
            print(sess.run(final_res))
