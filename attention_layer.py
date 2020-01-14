import tensorflow as tf
from tensorflow_core.python.keras import layers
from utils import common_layer

class Attention(layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self,hidden_size,num_heads,attention_dropout):
        """Initialize Attention.

        Args:
        hidden_size: int, output dim of hidden layer.
        num_heads: int, number of heads to repeat the same attention structure.
        attention_dropout: float, dropout rate inside attention for training.
        """
        if hidden_size%num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(hidden_size, num_heads))

        super().__init__()
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.attention_dropout=attention_dropout

    def build(self,input_shape):
        """Build the layer."""
        # Layers for linearly projecting the queries, keys, and values.
        size_per_head=self.hidden_size/self.num_heads
        self.query_dense_layer=common_layer.Dense3D(
            self.num_heads,size_per_head,'glorot_uniform',use_bias=False,name='query'
        )
        self.key_dense_layer=common_layer.Dense3D(
            self.num_heads,size_per_head,'glorot_uniform',use_bias=False,name='key'
        )
        self.value_dense_layer=common_layer.Dense3D(
            self.num_heads,size_per_head,'glorot_uniform',use_bias=False,name='value'
        )
        self.output_dense_layer=common_layer.Dense3D(
            self.num_heads,size_per_head,'glorot_uniform',use_bias=False,
            output_projection=True,name='output_transform'
        )
        super().build(input_shape)

    def get_config(self):
        return {
            'hidden_size':self.hidden_size,
            'num_heads':self.num_heads,
            'attention_dropout':self.attention_dropout
        }

    def call(self,query_input,source_input,bias,training,cache=None,decode_loop_step=None):
        """Apply attention mechanism to query_input and source_input.

        Args:
            query_input: [B , len_query , hidden_size]
            source_input: [B , len_souce , hidden_size ]
            bias: [B, 1, len_query, len_source]
            training: bool
            cache: (Used during prediction) A dictionary with tensors containing
                results of previous attentions. The dictionary must have the items:
                    {'k': tensor with shape [B,i,heads,dim_per_head],
                     'v': tensor with shape [B,i,heads,dim_per_head]}
                where i is the current decoded length for non-padded decode, or max
                sequence length for padded decode.
            decode_loop_step: An integer, step number of the decoding loop. Used only
                for autoregressive inference on TPU.

        Returns:
            Attention layer output with shape [B,len_query,hidden_size]
        """
        # Linearly project query, key and value using different learned
        # projections. Splitting heads is automatically done during the linear
        # projections --> [B, len, num_heads, dim_per_head]
        query=self.query_dense_layer(query_input)
        key=self.key_dense_layer(source_input)
        value=self.value_dense_layer(source_input)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            if decode_loop_step is not None:
                cache_k_shape=cache['k'].shape.as_list()
                indices=tf.reshape(
                    tf.one_hot(decode_loop_step,cache_k_shape[1],dtype=key.dtype),
                    [1,cache_k_shape[1],1,1]
                )
                key=cache['k']+key*indices
                cache_v_shape=cache['v'].shape.as_list()
                indices=tf.reshape(
                    tf.one_hot(decode_loop_step,cache_v_shape[1],dtype=value.dtype),
                    [1,cache_v_shape[1],1,1]
                )
                value=cache['v']+value*indices
            else:
                key=layers.concatenate([tf.cast(cache['k'],key.dtype),key],axis=1)
                value=layers.concatenate([tf.cast(cache['v'],value.dtype),key],axis=1)

            # Update cache
            cache['k']=key
            cache['v']=value

        # Scale query to prevent the dot product between query and key from growing too large.
        depth=(self.hidden_size//self.num_heads)
        query*=depth**-0.5

        # Calculate dot product attention
        logits=tf.einsum('BTNH,BFNH->BNFT',key,query)
        logits+=bias
        # Note that softmax internally performs math operations using float32
        # for numeric stability. When training with float16, we keep the input
        # and output in float16 for better performance.
        weights=layers.Softmax('attention_weights')(logits)
        if training:
            weights=layers.Dropout(self.attention_dropout)(weights)
        attention_output=tf.einsum('BNFT,BTNH->BFNH',weights,value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done --> [batch_size, length, hidden_size]
        attention_output=self.output_dense_layer(attention_output)
        return attention_output

class SelfAttention(Attention):
    """Mutiheaded self-attention layer."""

    def call(self,query_input,bias,training,cache=None,decode_loop_step=None):
        return super().call(query_input,query_input,bias,training,cache,decode_loop_step)