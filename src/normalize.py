import tensorflow as tf
from keras.layers.core import Layer

class normalize(Layer):
  def __init__(self, **kwargs) :
    super(normalize, self).__init__(**kwargs)

  def build(self, input_shape) :
    if len(input_shape) > 4 :
      raise Exception('Spatial Transformer must be called on a list of at most length 4 ')

    self.built = True

  def call(self, input) :
    return self._normalize(input)

  def _normalize(self, flow) :
    if len(flow.shape) == 4 :
      flow0 = flow[0, :, :, 0]
      flow1 = flow[0, :, :, 1]

      tf.reshape(flow0, [1024,1024])
      tf.reshape(flow1, [1024,1024])
        
      norm_flow0 = tf.div(tf.subtract(flow0, tf.reduce_min(flow0)), tf.subtract(tf.reduce_max(flow0), tf.reduce_min(flow0)))
      norm_flow1 = tf.div(tf.subtract(flow1, tf.reduce_min(flow1)), tf.subtract(tf.reduce_max(flow1), tf.reduce_min(flow1)))

      new_flow = tf.stack([norm_flow0, norm_flow1], axis = 2)
      new_flow = tf.reshape(new_flow, [1, 1024, 1024, 2])

      return new_flow
    '''
    else:
      flow0 = flow[0, :, :, 0]
      norm_flow0 = tf.div(tf.subtract(flow0, tf.reduce_min(flow0)), tf.subtract(tf.reduce_max(flow0), tf.reduce_min(flow0)))

      new_flow = tf.reshape(new_flow, [1, 1024, 1024, 1])

      return new_flow
    '''