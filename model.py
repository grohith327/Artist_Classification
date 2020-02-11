import os
import glob
import numpy as np
from scipy.io import wavfile
from sklearn.utils import shuffle
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras import Model

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(64, 2, padding='same', activation='relu')
    self.conv2 = Conv2D(128, 2, padding='same', activation='relu')
    self.conv3 = Conv2D(256, 2, padding='same', activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(4096, activation='relu')
    self.d2 = Dropout(0.4)
    self.d3 = Dense(2048, activation='relu')
    self.d4 = Dropout(0.4)
    self.d5 = Dense(1024, activation='relu')
    self.d6 = Dropout(0.4)
    self.d7 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.d5(x)
    x = self.d6(x)
    x = self.d7(x)
    return x

audio_files = []
audio_labels = []
count = 0
for dirc in sorted(os.listdir('data/')):
    if(count == 0):
        count+=1
        continue
    print("Reading ",dirc)
#     print(sorted(os.listdir('data/'+str(dirc))))
    for file in os.listdir('data/'+str(dirc)):
#         print(file)
        fs, data = wavfile.read('data/'+str(dirc)+'/'+str(file))
        audio_files.append(data)
        audio_labels.append(count-1)
    count += 1

audio_files = np.array(audio_files)
audio_labels = np.array(audio_labels)

audio_files = audio_files.reshape(2260,882000,2,1)

audio_files, audio_labels = shuffle(audio_files,audio_labels,random_state=0)

# Create an instance of the model
model = MyModel()

loss = tf.keras.metrics.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.RMSprop()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(audio_files, audio_labels):
  with tf.GradientTape() as tape:
    predictions = model(audio_files)
    loss = loss(audio_labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


audio_files = tf.cast(audio_files,tf.float32)

epochs = 5

for epoch in range(epochs):
    for audio_file, audio_label in zip(audio_files, audio_labels):
        audio_file = tf.reshape(audio_file,(1,882000, 2, 1))
#         print(audio_file)
        train_step(audio_file, audio_label)
    
    print("Epoch:"+epoch+' Loss:'+train_loss.result(),' Accuracy:'+train_accuracy.result()*100)