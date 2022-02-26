import tensorflow as tf
from tensorflow import keras
from keras import backend, initializers, regularizers, constraints, activations
from keras.engine.input_spec import InputSpec
from keras import layers
from keras.engine import base_layer

#--------------------------------------------------------------------
# Custom layers defined in https://arxiv.org/pdf/1912.04958.pdf
#--------------------------------------------------------------------

class BiasNoiseBroadcastLayer(tf.keras.layers.Layer):
    '''Combines an input x with noise image and an internal bias. Input must be [x, noise]'''

    def __init__(self, filter_size, *args, **kwargs):
        super(BiasNoiseBroadcastLayer, self).__init__(*args, **kwargs)
        self.filter_size = filter_size

    def build(self, input_shape):
        n, h, w, c = input_shape[0]

        assert (self.filter_size == c)       
        #initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)

        #bias for each feature map
        self.b = self.add_weight('kernel', shape=(1, 1, 1, c), initializer="zeros", trainable=True)

    def call(self, inputs):         
        x, noise = inputs
        return x + tf.multiply(self.b, noise)

    def get_config(self):
        config = super(BiasNoiseBroadcastLayer, self).get_config()
        config.update({"filter_size": self.filter_size})
        return config

#--------------------------------------------------------------------

class Conv2DMod(tf.keras.layers.Conv2D):
    ''' input [x, w],  w(latent) modulates x(conv input), w mods all batches and feature maps'''

    def __init__(self,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                groups=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                demod=True,
                **kwargs):
        super(Conv2DMod, self).__init__(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        dilation_rate=dilation_rate,
                        groups=groups,
                        activation=activations.get(activation),
                        use_bias=use_bias,
                        kernel_initializer=initializers.get(kernel_initializer),
                        bias_initializer=initializers.get(bias_initializer),
                        kernel_regularizer=regularizers.get(kernel_regularizer),
                        bias_regularizer=regularizers.get(bias_regularizer),
                        activity_regularizer=regularizers.get(activity_regularizer),
                        kernel_constraint=constraints.get(kernel_constraint),
                        bias_constraint=constraints.get(bias_constraint),
                        **kwargs)

        #set to 2 inputs, ndim is rank of tensor (or number of array dimensions)
        self.input_spec = [ InputSpec(ndim=self.rank + 2), InputSpec(min_ndim = 2)]
        self.demod = demod

    #copied from keras/layers/convolutional.py#L242 (22.02.2022; 10:27), only first line changed
    def call(self, inputs):
        #here the implementation has a problem, changes to account multiple inputs!!
        input_shape = inputs[0].shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))

        outputs = self.convolution_op(inputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return tf.nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = conv_utils.squeeze_batch_dims(outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format=self._tf_data_format)
        if not tf.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def build(self, input_shape):
        x_input_shape = input_shape[0]
        super(Conv2DMod, self).build(x_input_shape)

        #----------------------------------------------
        #define new convolution_op
        default_conv_op = self.convolution_op  
        
        def conv2d_mod_op(inputs, kernel):
            # x = inputs[0]   has shape (batches, w, h, channels)
            # kernel w tensor has shape (rows, cols, input_depth, output_depth), s_i is the modulation of the ith input feature
            # w_ijk = s_i * w_ijk,    j and k enumerate the output feature maps and spatial footprint of the convolution
            # w = kernel            
            #kernel = tf.tensordot(self.latent_inject, kernel, axis=())
            
            #Add batch layer to kernel: (1, rows k, cols k, input_depth i, output_depth j)
            kernel = backend.expand_dims(kernel, axis = 0)

            # Change s to (batch_size, 1, 1, scales=input_depth, 1)
            s = backend.expand_dims(backend.expand_dims(backend.expand_dims(inputs[1], axis = 1), axis = 1), axis = -1)

            #------------------------
            #Modulate, gives (batch_size, rows, cols, input_depth, output_depth)
            kernel = tf.multiply(kernel, s+1)

            #------------------------
            #Demodulate, scale ("demodulate") each output feature map j
            if self.demod:
                d = backend.epsilon() + backend.sqrt(tf.reduce_sum(tf.square(kernel), axis=[1, 2, 3], keepdims=True))
                kernel = kernel / d

            #------------------------            
            #Fuse
            x = tf.transpose(inputs[0], [0, 3, 1, 2])                   #[BHWC] -> [BCHW]
            x = tf.reshape(x, [1, -1, tf.shape(x)[2], tf.shape(x)[3]])  #[1, channels * batches, h, w]
            x = tf.transpose(x, [0, 2, 3, 1])                           #[1, h, w, channels * batches]

            w = tf.transpose(kernel, [1, 2, 3, 0, 4])                                   #[rows, cols, input_depth, batches, output_depth]
            w = tf.reshape(w, [kernel.shape[1], kernel.shape[2], kernel.shape[3], -1])  #[rows, cols, input_depth, batches*output_depth]

            #------------------------        
            #Convolute
            output = default_conv_op(x, w)  #[1, new_h, new_w, batches*output_depth]      
            
            #------------------------
            #Unfuse
            output = tf.transpose(output, [0, 3, 1, 2])                                                 #[1, batches*output_depth, new_h, new_w] 
            output = tf.reshape(output, [-1, self.filters, tf.shape(output)[2], tf.shape(output)[3]])   #[batches, output_depth, new_h, new_w]
            output = tf.transpose(output, [0, 2, 3, 1])                                                 #[batches, new_h, new_w, batches*output_depth]           
            return output
                
        #----------------------------------------------

        #set inputs
        input_shape = tf.TensorShape(x_input_shape)
        input_channel = self._get_input_channel(input_shape)
        channel_axis = self._get_channel_axis()
        self.input_spec = [InputSpec(ndim=self.rank + 2, axes={channel_axis: input_channel}),  InputSpec(ndim = 2)] 

        #set new convolution_op and finish build
        self.convolution_op = conv2d_mod_op
        self.built = True

    def get_config(self):
        config = super(Conv2DMod, self).get_config()
        config.update({"demod": self.demod})
        return config

#--------------------------------------------------------------------