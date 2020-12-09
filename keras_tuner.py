# -*- coding: utf-8 -*-

!pip uninstall tensorflow

!pip install tensorflow-gpu==2.0.0

!pip install keras-tuner==1.0.0 --no-dependencies

!pip install terminaltables colorama

import tensorflow as tf
import kerastuner as kt
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) =\
    tf.keras.datasets.fashion_mnist.load_data()

import numpy as np
train_images = np.asarray(train_images, dtype=np.float32) / 255.0
test_images = np.asarray(test_images, dtype=np.float32) / 255.0
train_images = train_images.reshape(60000,784)
test_images = test_images.reshape(10000,784)

train_labels = tf.keras.utils.to_categorical(train_labels)

dataset = tf.data.Dataset.from_tensor_slices((train_images.astype(np.float32),
                                              train_labels.astype(np.float32)))

k = 0
for i in dataset:
  print(i)
  if k==2:
    break
  k+=1

!rm -rf results

class my_layer(tf.keras.layers.Layer):
  
  def __init__(self,ou):
    super(my_layer,self).__init__()
    
    
    self.W = tf.Variable(initial_value=np.random.randn(784,ou),trainable = True, dtype = tf.float32)
  def call(self,inputs):
    
    return tf.nn.relu(tf.matmul(inputs,self.W))

def build_model(hp):
  """Builds a dnn model."""
  class mod(tf.keras.Model):
    def __init__(self):
      super(mod,self).__init__()
      ou = hp.Int('nou', 120, 200, step=20, default=120)
      ou1 = hp.Int('nou_w', 120, 200, step=20, default=120)
      self.layer0 = my_layer(ou1)
      self.layer1 = tf.keras.layers.Dense(ou,activation = tf.nn.relu)
      self.layer2 = tf.keras.layers.Dense(10,activation = tf.nn.softmax)
    def call(self,inputs):
      return self.layer2(self.layer1(self.layer0(inputs)))
  model = mod()
  
  return model

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=200):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

temp_learning_rate_schedule = CustomSchedule(512)

plt.plot(temp_learning_rate_schedule(tf.range(4000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")

def get_loss(lab,logits):
  return tf.keras.losses.categorical_crossentropy(lab,logits)

class MyTuner(kt.Tuner):

    def run_trial(self, trial, train_ds):
        hp = trial.hyperparameters
        
        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        
        train_ds = train_ds.batch(
            hp.Int('batch_size', 32, 128, step=32, default=64))
        
        model = self.hypermodel.build(trial.hyperparameters)
        lr = CustomSchedule(512)
        #lr = 0.001
        optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
        epoch_loss_metric = tf.keras.metrics.Mean()
        
        @tf.function
        def run_train_step(images,lab):
            with tf.GradientTape() as t:
              logits = model(images)
              loss = tf.reduce_mean(get_loss(lab,logits))
              grads = t.gradient(loss, model.trainable_variables)
      
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            return loss
        # `self.on_epoch_end` reports results to the `Oracle` and saves the
        # current state of the Model. The other hooks called here only log values
        # for display but can also be overridden. For use cases where there is no
        # natural concept of epoch, you do not have to call any of these hooks. In
        # this case you should instead call `self.oracle.update_trial` and
        # `self.oracle.save_model` manually.
        for epoch in range(5):
            print('Epoch: {}'.format(epoch))

            #self.on_epoch_begin(trial, model, epoch, logs={})
            for (batch, (images, lab)) in enumerate(train_ds):
                
                batch_loss = float(run_train_step(images,lab))
                

                if batch % 100 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print('Batch: {}, Average Loss: {}'.format(batch, loss))

            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epoch, logs={'fin_loss': epoch_loss})
            epoch_loss_metric.reset_states()

tuner = MyTuner(
oracle=kt.oracles.BayesianOptimization(
          objective=kt.Objective('fin_loss', 'min'),
          max_trials=10),
      hypermodel=build_model,
      directory='results',
      project_name='mnist_custom_training')


tuner.search(train_ds=dataset)

m = tuner.get_best_models()[0]

k = 0
for i in dataset:
  #print(i[1])
  print('REAL -> {}'.format(np.argmax(i[1])))
  pred = m(i[0][np.newaxis,:])
  print('PRED -> {}'.format(np.argmax(pred,axis = 1)))
  k+=1
  if k==50:
    break

best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)

for i in tuner.get_best_hyperparameters(num_trials = 10):
  print(i.values)

tuner.results_summary()

