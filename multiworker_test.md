Multi-Worker Mirrored Strategy
================

This script is just a test using tf.distributed with multiworker
strategy. This code worked in my local ubuntu 18.04, “locally” in the
master machine in EMR and in the EMR machine using the workers ip’s.

``` python
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
```

for multiworker the cluster, worker and task need to be store in a
enviroment variable named TF\_CONFIG and must be declared before
strategy.

``` python
os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
    'worker': ["10.16.90.112:12345", "10.28.201.223:45282", "10.45.177.250:57780"]
    # # "10.16.203.189:56936" one of the machines
  },
  'task': {'type': 'worker', 'index': 0}
})
```

here is a bug in multiworker. Strategy needs to be the first command
when using TF. This prevent the error: RuntimeError: Collective ops must
be configured at program startup
<https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras>

``` python
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```

Here is just preparing the data for trainning

``` python
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
```

Just a function for build and compile the CNN

``` python
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
```

Define parameters

``` python
NUM_WORKERS = 2
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
```

Run the scope

``` python
with strategy.scope():
  train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
  multi_worker_model = build_and_compile_cnn_model()
```

Finally :)

``` python
multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)
```

    Train for 5 steps
    Epoch 1/3
    5/5 [==============================] - 3s 519ms/step - loss: 2.3055 - accuracy: 0.1109
    Epoch 2/3
    5/5 [==============================] - 0s 29ms/step - loss: 2.2941 - accuracy: 0.1250
    Epoch 3/3
    5/5 [==============================] - 0s 29ms/step - loss: 2.2903 - accuracy: 0.1187
    <tensorflow.python.keras.callbacks.History object at 0x7f21ac3e0350>

### Other workers

I’m running this in `10.28.201.223`

``` python
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
    'worker': ["10.16.90.112:12345", "10.28.201.223:45282", "10.45.177.250:57780"]
    # # "10.16.203.189:56936" one of the machines
  },
  'task': {'type': 'worker', 'index': 1}
})
```

and this in `10.45.177.250`

``` python
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ['TF_CONFIG'] = json.dumps({
  'cluster': {
    'worker': ["10.16.90.112:12345", "10.28.201.223:45282", "10.45.177.250:57780"]
    # # "10.16.203.189:56936" one of the machines
  },
  'task': {'type': 'worker', 'index': 2}
})
```
