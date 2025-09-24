import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight("att_W", (input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight("att_b", (input_shape[-1],), initializer="zeros", trainable=True)
        self.u = self.add_weight("att_u", (input_shape[-1],), initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait)
        a = tf.expand_dims(a, -1)
        return tf.reduce_sum(x * a, axis=1)
