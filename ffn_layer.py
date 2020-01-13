import tensorflow as tf
from tensorflow_core.python.keras import layers,activations

class FeedForwardNetword(layers.Layer):

    def __init__(self,hidden_size,filter_size,relu_dropout):
        super().__init__()
        self.hidden_size=hidden_size
        self.filter_size=filter_size
        self.relu_dropout=relu_dropout

    def build(self,input_shape):
        self.filter_dense_layer=layers.Dense(
            self.filter_size,
            use_bias=True,
            activation=activations.relu,
            name='filter_layer'
        )
        self.output_dense_layer=layers.Dense(
            self.hidden_size,use_bias=True,name='output_layer'
        )
        super().build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }

    def call(self,x,training):
        """Return outputs of the feedforward network.

        Args:
        x: tensor with shape [batch_size, length, hidden_size]
        training: boolean, whether in training mode or not.

        Returns:
        Output of the feedforward network.
        tensor with shape [batch_size, length, hidden_size]
        """
        output=self.filter_dense_layer(x)
        if training:
            output=layers.Dropout(self.relu_dropout)(output)
        output=self.output_dense_layer(output)

        return output