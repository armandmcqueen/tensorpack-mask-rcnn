

import tensorflow as tf


# Checks at graph_build time
def print_buildtime_shape(name, tensor):
    print(f'[buildtime_shape] {name}: {tensor.shape}')



# There is both a tensor and a trigger tensor so that you can
def print_runtime_shape(name, tensor):
    s = "[runtime_shape] "+name+": "+str(tf.shape(tensor))
    return runtime_print(s, tensor)



# A method if you want tf.print to behave like tf.Print (i.e. the 'print' exists as an op in the computation graph)
"""
some_tensor = tf.op(some_other_tensor)
some_tensor = runtime_print("String to print", some_tensor)
"""
def runtime_print(message, trigger_tensor):
    print_op = tf.print(message)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)