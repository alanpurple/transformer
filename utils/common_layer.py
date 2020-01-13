import tensorflow as tf
from tensorflow_core.python.keras import layers

class Dense3D(layers.Layer):
    """A Dense Layer using 3D kernel with tf.einsum implementation.

    Attributes:
        num_attention_heads: An integer, number of attention heads for each
            multihead attention layer.
        size_per_head: An integer, hidden size per attention head.
        hidden_size: An integer, dimension of the hidden layer.
        kernel_initializer:
        bias_initializer:
        activation:
        use_bias:
        output_projection: A bool, whether the Dense3D layer is used for output
            linear projection
    """

    def __init__(self,num_attention_heads=12,size_per_head=72,kernel_initializer=None,
                 bias_initializer='zeros',activation=None,use_bias=True,output_projection=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_attention_heads=num_attention_heads
        self.size_per_head=size_per_head
        self.hidden_size=num_attention_heads*size_per_head
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.activation=activation
        self.use_bias=use_bias
        self.output_projection=output_projection

    @property
    def compatible_kernel_shape(self):
        if self.output_projection:
            return [self.hidden_size, self.hidden_size]
        return [self.last_dim,self.hidden_size]

    @property
    def compatible_bias_shape(self):
        return [self.hidden_size]

    @property
    def kernel_shape(self):
        if self.output_projection:
            return [self.num_attention_heads,self.size_per_head,self.hidden_size]
        return [self.last_dim,self.num_attention_heads,self.size_per_head]

    @property
    def bias_shape(self):
        if self.output_projection:
            return [self.hidden_size]
        return [self.num_attention_heads,self.size_per_head]
    
    def build(self,input_shape):
        dtype=tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `Dense3D` layer with non-floating "
                            "point (and non-complex) dtype %s" % (dtype,))
        input_shape=tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to `Dense3D` "
                             "should be defined. Found `None`.")
        self.last_dim=tf.compat.dimension_value(input_shape[-1])
        self.input_spec=layers.InputSpec(min_ndim=3,axes={-1: self.last_dim})

        self.kernel=self.add_weight(
            'kernel',
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            self.bias=self.add_weight(
                'bias',
                shape=self.bias_shape,
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias=None
        super().build(input_shape)

    def call(self,inputs):
        """Implements   ``call()`` for Dense3D.

        Args:
            inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
            when output_projection is False, otherwise a float tensorf of shape
            [batch_size, sequence_length, num_heads, dim_per_head].

        Returns:
            The projected tensor with shape [batch_size, sequence_length, num_heads,
            dim_per_head] when output_projection is False, otherwise [batch_size,
            sequence_length, hidden_size].
        """

        if self.output_projection:
            ret=tf.einsum('abcd,cde->abe',inputs,self.kernel)
        else:
            ret=tf.einsum('abc,cde->abde',inputs,self.kernel)
        if self.use_bias:
            ret+=self.bias
        if self.activation is not None:
            ret=self.activation(ret)
        return ret