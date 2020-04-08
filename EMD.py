import tensorflow.compat.v1 as tf

tf.executing_eagerly()

a = [[4, 3, 1], [10, 9, 1]]
b = tf.sort(a, direction='ASCENDING', name=None)
c = tf.keras.backend.eval(b)
# Here, c = [  1.     2.8   10.    26.9   62.3  166.32]
print(c)
