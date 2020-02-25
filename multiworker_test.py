import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds

# here is a bug in multiworker. Strategy needs to be the first command when using TF.
# This prevent the error: RuntimeError: Collective ops must be configured at program startup
# https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# for multiworker the cluster, worker and task need to be store in a enviroment variable
# named TF_CONFIG

os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
    'worker': ["localhost:12345", "localhost:23456"]
    # or your ip's ex.: ["11.22.333.444:12345", "22.33.444.555:12345"]
  },
  'task': {'type': 'worker', 'index': 0}
})

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def make_datasets_unbatched():
  # Scaling MNIST data from (0, 255] to (0., 1.]
 def scale(image, label):
   image = tf.cast(image, tf.float32)
   image /= 255
   return image, label
    
 datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            as_supervised=True)
 return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)
  

train_datasets = make_datasets_unbatched().batch(BATCH_SIZE)

# just a function for build and compipe the CNN
def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics = ['accuracy'])
  return model

# define parameters
NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS

# run the scope
with strategy.scope():
  train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
  multi_worker_model = build_and_compile_cnn_model()

# finally :)
multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)

