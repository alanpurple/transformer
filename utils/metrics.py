import tensorflow as tf
from tensorflow_core.python.keras import losses,layers
from tensorflow_core.python.keras.metrics import Mean
import numpy as np
import collections
import math
import functools

def _get_ngrams_with_counter(segment,max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.

    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this methods.

    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngrams_counts=collections.Counter()
    for order in range(1,max_order+1):
        for i in range(len(segment)-order+1):
            ngram=tuple(segment[i:i+order])
            ngrams_counts[ngram]+=1
    return ngrams_counts


def compute_bleu(reference_corpus,translation_corpus,max_order=4,use_bp=True):
    reference_length=0
    translation_length=0
    geo_mean=0

    matches_by_order=[0]*max_order
    possible_matches_by_order=[0]*max_order
    precisions=[]

    for (references,translations) in zip(reference_corpus,translation_corpus):
        reference_length+=len(references)
        translation_length+=len(translations)
        ref_ngram_counts=_get_ngrams_with_counter(references,max)
        translation_ngram_counts=_get_ngrams_with_counter(translations,max_order)

        overlap=dict((ngram,min(count,translation_ngram_counts[ngram]))
                    for ngram,count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram)-1]+=overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram)-1]+=translation_ngram_counts[ngram]

    precisions=[0]*max_order
    smooth=1.0

    for i in range(max_order):
        if possible_matches_by_order[i]>0:
            if matches_by_order[i]>0:
                precisions[i]=float(matches_by_order[i])/possible_matches_by_order[i]
            else:
                smooth*=2
                precisions[i]=1.0/(smooth*possible_matches_by_order[i])
        else:
            precisions[i]=0.0

    if max(precisions)>0:
        p_log_sum=sum(math.log(p) for p in precisions if p)
        geo_mean=math.exp(p_log_sum/max_order)
    
    if use_bp:
        ratio=translation_length/reference_length
        bp=math.exp(1 - 1. /ratio) if ratio<1.0 else 1.0
        return np.float32(geo_mean*bp)
    else:
        return np.float32(geo_mean)

def _pad_tensors_to_same_length(x,y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope('pad_to_same_length'):
        x_len=tf.shape(x)[1]
        y_len=tf.shape(y)[1]

        max_len=tf.maximum(x_len,y_len)

        x=tf.pad(x,[[0,0],[0,max_len-x_len],[0,0]])
        y=tf.pad(y,[[0,0],[0,max_len-y_len]])
    return x,y

def padded_cross_entropy_loss(logits,labels,smoothing,vocab_size):
    """Calculate cross entropy loss while ignoring padding.

    Args:
        logits: [B,len_logits,vocab_size]
        labels: [B,len_labels]
        smoothing: Label smoothing constant
        vocab_size: vocab

    Returns:
        Returns the cross entropy loss and weight tensors: float32 tensors with
         shape [B,max(len_logits,len_lables)]
    """
    with tf.name_scope('loss'):
        logits,labels=_pad_tensors_to_same_length(logits,labels)

        with tf.name_scope('smoothing cross entropy'):
            confidence=1.0-smoothing
            low_confidence=(1.0-confidence)/tf.cast(vocab_size-1,tf.float32)
            soft_targets=tf.one_hot(tf.cast(labels,tf.int32),depth=vocab_size,
                                    on_value=confidence,off_value=low_confidence)
            xentropy=losses.CategoricalCrossentropy(True)(soft_targets,logits)

            normalizing_constant=-(
                confidence*tf.math.log(confidence)+
                tf.cast(vocab_size-1,tf.float32)*low_confidence*tf.math.log(low_confidence+1e-20)
                )
            xentropy -= normalizing_constant

    weights=tf.cast(tf.not_equal(labels,0),tf.float32)
    return xentropy * weights, weights

def padded_accuracy(logits, labels):
  """Percentage of times that predictions matches labels on non-0s."""
  with tf.name_scope("padded_accuracy"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
    outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    padded_labels = tf.cast(labels, tf.int32)
    return tf.cast(tf.equal(outputs, padded_labels), tf.float32), weights


def padded_accuracy_topk(logits, labels, k):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.name_scope("padded_accuracy_topk"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
    effective_k = tf.minimum(k, tf.shape(logits)[-1])
    _, outputs = tf.nn.top_k(logits, k=effective_k)
    outputs = tf.cast(outputs, tf.int32)
    padded_labels = tf.cast(labels, tf.int32)
    padded_labels = tf.expand_dims(padded_labels, axis=-1)
    padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
    same = tf.cast(tf.equal(outputs, padded_labels), tf.float32)
    same_topk = tf.reduce_sum(same, axis=-1)
    return same_topk, weights


def padded_accuracy_top5(logits, labels):
  return padded_accuracy_topk(logits, labels, 5)


def padded_sequence_accuracy(logits, labels):
  """Percentage of times that predictions matches labels everywhere (non-0)."""
  with tf.name_scope("padded_sequence_accuracy"):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
    outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    padded_labels = tf.cast(labels, tf.int32)
    not_correct = tf.cast(tf.not_equal(outputs, padded_labels),
                          tf.float32) * weights
    axis = list(range(1, len(outputs.get_shape())))
    correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
    return correct_seq, tf.constant(1.0)


def padded_neg_log_perplexity(logits, labels, vocab_size):
  """Average log-perplexity excluding padding 0s. No smoothing."""
  num, den = padded_cross_entropy_loss(logits, labels, 0, vocab_size)
  return -num, den

class MetricLayer(layers.Layer):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.metric_mean_fns=[]

    def build(self,input_shape):
        neg_log_perplexity=functools.partial(
            padded_neg_log_perplexity,vocab_size=self.vocab_size
        )
        self.metric_mean_fns=[
            (Mean('accuracy'),padded_accuracy),
            (Mean('accuracy_top5'),padded_accuracy_top5),
            (Mean('accuracy_per_sequence'),padded_sequence_accuracy),
            (Mean('neg_log_perplexity'),neg_log_perplexity)
        ]
        super().build(input_shape)

    def get_config(self):
        return {'vocab_size':self.vocab_size}

    def call(self,inputs):
        logits,targets=inputs[0],inputs[1]
        for mean,fn in self.metric_mean_fns:
            m=mean(*fn(logits,targets))
            self.add_metric(m)
        return logits

def transformer_loss(logits,labels,smoothing,vocab_size):
    """Calculates total loss containing cross entropy with padding ignored.

    Args:
        logits: [B,len_logits,vocab_size]
        labels: [B,len_labels]
        smoothing: :Label smoothing constant
        vocab_size: vocab

    Returns:
        A scalar float tensor for loss.
    """
    xentropy,weights=padded_cross_entropy_loss(logits,labels,smoothing,vocab_size)
    return tf.reduce_sum(xentropy)/tf.reduce_sum(weights)