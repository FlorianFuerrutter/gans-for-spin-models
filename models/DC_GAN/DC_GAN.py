import numpy as np
import matplotlib.pyplot as plt

#Tensorflow
import tensorflow as tf
print("[INFO] Using tf version: " + tf.__version__ )
print("[INFO] Build with gpu support: " + format(tf.test.is_built_with_gpu_support()))

device_name = tf.test.gpu_device_name()  
if device_name != '/device:GPU:0':
   print("[WARNING] GPU device not found, using CPU for inference")
   print("  -> TensorFlow GPU support requires an assortment of drivers and libraries: https://www.tensorflow.org/install/gpu#software_requirements")
else:
    print('[INFO] Found GPU at: {}'.format(device_name))

#-------------------------------------
#-------------------------------------

def main() -> int:   
    return 0


if __name__ == '__main__':
    main()