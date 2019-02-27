

import tensorflow as tf


# Checks at graph_build time
def print_buildtime_shape(name, tensor, prefix=None):
    if prefix is not None:
        prefix = f' [{prefix}]'
    else:
        prefix = ""

    print(f'[buildtime_shape]{prefix} {name}: {tensor.shape}')




def print_runtime_shape(name, tensor, prefix=None):
    s = "[runtime_shape] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += f'{name}: '
    return runtime_print([s, tf.shape(tensor)], tensor)



# A method if you want tf.print to behave like tf.Print (i.e. the 'print' exists as an op in the computation graph)
"""
some_tensor = tf.op(some_other_tensor)
some_tensor = runtime_print("String to print", some_tensor)
"""
def runtime_print(message, trigger_tensor):
    print_op = tf.print(message)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)


def runtime_print_str(message_str, trigger_tensor, prefix=None):
    if prefix is not None:
        message_str = f'[{prefix}] {message_str}'

    return runtime_print(message_str, trigger_tensor)




def print_runtime_tensor(name, tensor, prefix=None, summarize=-1):
    s = "[runtime_tensor] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += name

    print_op = tf.print(s, tensor, summarize=summarize)
    with tf.control_dependencies([print_op]):
        return tf.identity(tensor)


def print_runtime_tensor_loose_branch(name, tensor, prefix=None, summarize=-1, trigger_tensor=None):
    assert trigger_tensor is not None

    s = "[runtime_tensor_freehanging_branch] "
    if prefix is not None:
        s += f'[{prefix}] '
    s += name

    print_op = tf.print(s, tensor, summarize=summarize)
    with tf.control_dependencies([print_op]):
        return tf.identity(trigger_tensor)