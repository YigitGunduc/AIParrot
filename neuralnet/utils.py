def check_cuda():
    '''
    checks if cuda is available or not
    '''
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        return False 
    return True
