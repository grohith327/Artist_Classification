import numpy as np 
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
import librosa
import os
from sklearn.utils import shuffle



@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

class MusicModel(Model):
    def __init__(self):
        super(MusicModel,self).__init__()
        self.fc1=Dense(10, activation='relu')
        self.fc2=Dropout(0.4)
        self.fc3=Dense(10, activation='relu')
        self.fc4=Dropout(0.4)
        self.fc5=Dense(5, activation='softmax')
    def call(self,x):
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        x=self.fc5(x)
        return x

# audio_features =np.load('audio_tracks_3_sec.npy')
# audio_labels=np.load('audio_labels_3_sec.npy') 

audio_features = []
audio_labels = []

count = 0
for artist in os.listdir('trim_data_3_sec'):
    print('Reading '+str(artist))
    for tracks in os.listdir('trim_data_3_sec/'+str(artist)):
        data, sr = librosa.load('trim_data_3_sec/'+str(artist)+'/'+str(tracks))
        audio_features.append(data)
        audio_labels.append(count)
    count += 1

audio_features = np.array(audio_features)
audio_labels = np.array(audio_labels)


X_data=audio_features
Y_data=audio_labels

# v_min = audio_features.min(axis=(1,2), keepdims=True)
# v_max = audio_features.max(axis=(1,2), keepdims=True)
# X_data=(audio_features - v_min)/(v_max - v_min)

print(X_data.shape, Y_data.shape)

X_data, Y_data = shuffle(X_data, Y_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2,stratify=Y_data,random_state=42)

y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

# X_train = X_train.reshape((-1,216,43,1))
# X_test = X_test.reshape((-1,216,43,1))

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
enc = OneHotEncoder()
y_test = enc.fit_transform(y_test).toarray()




train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

model=MusicModel()

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

EPOCHS=30
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()




