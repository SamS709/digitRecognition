import tensorflow as tf
import numpy as np
a = (tf.random.uniform([1,1])>0.7)
b = tf.cast(a,dtype="float32")
print(a)
print(b)
print(tf.cast((tf.random.uniform([1,1])>0.7),dtype="float32"))
L = [[1,1,1],[2,2]]
print(sum([2,2]))
print(np.concatenate(L).mean())