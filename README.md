tf.distributed
================

`tf.distribute.Strategy` is a TensorFlow API to distribute training
across multiple GPUs, multiple machines or TPUs. Using this API, users
can distribute their existing models and training code with minimal code
changes. `tf.distribute.Strategy` has been designed with these key goals
in mind:

  - Easy to use and support multiple user segments, including
    researchers, ML engineers, etc.
  - Provide good performance out of the box.
  - Easy switching between strategies.

`tf.distribute.Strategy` can be used with TensorFlow’s high level APIs,
`tf.keras` and `tf.estimator`, with just a couple of lines of code
change. It also provides an API that can be used to distribute custom
training loops (and in general any computation using TensorFlow). In
TensorFlow 2.0, users can execute their programs eagerly, or in a graph
using `tf.function`. `tf.distribute.Strategy` intends to support both
these modes of execution. Note that we may talk about training most of
the time in this guide, but this API can also be used for distributing
evaluation and prediction on different platforms.

# Strategies supported by keras

There are 5 strategies supported by keras API:

### Mirrored Strategy (Supported)

`tf.distribute.MirroredStrategy` supports synchronous distributed
training on multiple GPUs on one machine. It creates one replica per GPU
device. Each variable in the model is mirrored across all the replicas.
Together, these variables form a single conceptual variable called
MirroredVariable. These variables are kept in sync with each other by
applying identical updates.

### Central Storage Strategy (Experimental support)

`tf.distribute.experimental.CentralStorageStrategy` does synchronous
training as well. Variables are not mirrored, instead they are placed on
the CPU and operations are replicated across all local GPUs. If there is
only one GPU, all variables and operations will be placed on that GPU.

### MultiWorker Mirrored Strategy (Experimental support)

`tf.distribute.experimental.MultiWorkerMirroredStrategy` is very similar
to MirroredStrategy. It implements synchronous distributed training
across multiple workers, each with potentially multiple GPUs. Similar to
MirroredStrategy, it creates copies of all variables in the model on
each device across all workers.

It uses CollectiveOps as the multi-worker all-reduce communication
method used to keep variables in sync. A collective op is a single op in
the TensorFlow graph which can automatically choose an all-reduce
algorithm in the TensorFlow runtime according to hardware, network
topology and tensor sizes.

### Parameter Server Strategy (Support planned in tf 2.0 RC)

`tf.distribute.experimental.ParameterServerStrategy` supports parameter
servers training on multiple machines. In this setup, some machines are
designated as workers and some as parameter servers. Each variable of
the model is placed on one parameter server. Computation is replicated
across all GPUs of all the workers.

### TPU Strategy (Support planned in tf 2.0 RC)

`tf.distribute.experimental.TPUStrategy` lets users run their TensorFlow
training on Tensor Processing Units (TPUs). TPUs are Google’s
specialized ASICs designed to dramatically accelerate machine learning
workloads. They are available on Google Colab, the TensorFlow Research
Cloud and Google Compute Engine.

In terms of distributed training architecture, TPUStrategy is the same
MirroredStrategy - it implements synchronous distributed training. TPUs
provide their own implementation of efficient all-reduce and other
collective operations across multiple TPU cores, which are used in
TPUStrategy.

# Using strategies

To start, we need to load keras and reticulate package.

``` r
library(keras)
library(reticulate)
```

For each strategy you have to create an object. I’ll use two of them to
ilustrate.

``` r
tf <- import("tensorflow")

strategy <- list()

# Creating strategies
strategy$mirrored <- tf$compat$v1$distribute$MirroredStrategy()
strategy$central_storage <- tf$distribute$experimental$CentralStorageStrategy()
strategy$multiworker_mirrored <- tf$distribute$experimental$MultiWorkerMirroredStrategy()
strategy$parameter_server<-  tf$distribute$experimental$ParameterServerStrategy()
```

Let’s run a model to see how it works. I’ll the well known mnist dataset
as example.

``` r
# mnist dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
```

To use a strategy in a model you just need to call the method `scope()`
using `wtih`…

``` r
model <- keras_model_sequential()
with(strategy$central_storage$scope(), {
  model %>%
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')
})
```

…run\!

``` r
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10, batch_size = 128,
  validation_split = 0.2
)
```

To change the strategy, you just need to change the scope. For example

``` r
model <- keras_model_sequential()
with(strategy$parameter_server$scope(), {
  model %>%
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')
})
```
