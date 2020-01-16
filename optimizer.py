import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras import callbacks
from tensorflow_core.python.keras import backend as K

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self,initial_learning_rate,hidden_size,warmup_steps):
        super().__init__()
        self.initial_learning_rate=initial_learning_rate
        self.hidden_size=hidden_size
        self.warmup_steps=tf.cast(warmup_steps,tf.float32)

    def __call__(self,global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.

        Args:
            global_step: An integer, the current global step used for learning rate
                calculation.

        Returns:
            A float, the learning rate needs to be used for current global step.
        """
        with tf.name_scope('learning_rate_schedule'):
            global_step=tf.cast(global_step,tf.float32)
            learning_rate=self.initial_learning_rate
            learning_rate*=(self.hidden_size*-0.5)
            learning_rate*=tf.minimum(1.0,global_step/self.warmup_steps)
            learning_rate/=tf.sqrt(tf.maximum(global_step,self.warmup_steps))
        return learning_rate

    def get_config(self):
        return {
            'initial_learning_rate':self.initial_learning_rate,
            'hidden_size':self.hidden_size,
            'warmup_steps':self.warmup_steps
        }

class LearningRateFn(object):

    def __init__(self,learning_rate,hidden_size,warmup_steps):
        self.learning_rate=learning_rate
        self.hidden_size=hidden_size
        self.warmup_steps=float(warmup_steps)

    def __call__(self,global_step):
        step=float(global_step)
        learning_rate=self.learning_rate
        learning_rate*=(self.hidden_size**-0.5)
        learning_rate*=np.minimum(1.0,step/self.warmup_steps)
        learning_rate/=np.sqrt(np.maximum(step,self.warmup_steps))
        return learning_rate

class LearningRateScheduler(callbacks.Callback):

    def __init__(self,schedule,init_steps=None,verbose=False):
        super().__init__()
        self.schedule=schedule
        self.verbose=verbose
        if init_steps is None:
            init_steps=0.0
        self.steps=float(init_steps)
    
    def on_epoch_begin(self,epoch,logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not hasattr(self.model.optimizer, 'iterations'):
            raise ValueError('Optimizer must have a "iterations" attribute.')

    def on_train_batch_begin(self,batch,logs=None):
        if self.verbose>0:
            iterations=K.get_value(self.model.optimizer.iterations)
            print('Original iteration %d' % iterations)

        self.steps+=1.0
        lr=float(K.get_value(self.model.optimizer.lr))
        lr=self.schedule(self.steps,lr)
        if not isinstance(lr,(float,np.float32,np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr,lr)
        K.set_value(self.model.optimizer.iterations,self.steps)

        if self.verbose>0:
            print('Batch %05d Step %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (batch + 1, self.steps, lr))

    def on_epoch_end(self,epoch,logs=None):
        logs=logs or {}
        logs['lr']=K.get_value(self.model.optimizer.lr)
        logs['stpes']=self.steps