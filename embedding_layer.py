import tensorflow as tf
from tensorflow_core.python.keras import layers,initializers

class EmbeddingSharedWeights(layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self,vocab_size,hidden_size):
        """Specify characteristic parameters of embedding layer.

        Args:
            vocab_size: ~32,000
            hidden_size: dimensionality (Typically 512 or 1024)
        """
        super().__init__()
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size

    def build(self,input_shape):
        with tf.name_scope('embedding_and_softmax'):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.shared_weights=self.add_weight(
                'weight',
                shape=[self.vocab_size,self.hidden_size],
                initializer=initializers.RandomNormal(0.,self.hidden_size**-0.5)
            )
        super().build(input_shape)

    def get_config(self):
        return {
            'vocab_size':self.vocab_size,
            'hidden_size':self.hidden_size
        }

    def call(self,inputs,mode='embedding'):
        """Get token embeddings of inputs.

        Args:
        inputs: An int64 tensor with shape [batch_size, length]
        mode: string, a valid value is one of "embedding" and "linear".
        Returns:
        outputs: (1) If mode == "embedding", output embedding tensor, float32 with
            shape [batch_size, length, embedding_size]; (2) mode == "linear", output
            linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
        ValueError: if mode is not valid.
        """
        if mode=='embedding':
            return self._embedding(inputs)
        elif mode=='linear':
            return self._linear(inputs)
        else:
            raise ValueError('mode {} is not valid'.format(mode))

    def _embedding(self,inputs):
        with tf.name_scope('embedding'):
            embeddings=tf.gather(self.shared_weights,inputs)
            mask=tf.cast(tf.not_equal(inputs,0),embeddings.dtype)
            embeddings*=tf.expand_dims(mask,-1)
            embeddings*=self.hidden_size**0.5

        return embeddings

    def _linear(self,inputs):
        """Computes logits by running inputs through a linear layer.

        Args:
        inputs: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
        float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope('presoftmax_linear'):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])