import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle 
import numpy as np

img_size = 224 * 224 *1
with open('images.pkl','rb') as f:
  images = pickle.load(f)
 
with open('labels.pkl','rb') as f:
  labels = pickle.load(f)

#images = images.reshape(-1,img_size)

images = images/255.0

print(tf.config.experimental.list_physical_devices('GPU'))
 
epochs = 10
 
X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size = 0.3,random_state = 10)

encoding_dim = 32
batch_size = 128

input_signal = Input(shape=(224, 224, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_signal)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

encoded2 = Flatten()(encoded)
encoded2 = Dense(encoding_dim, activation='sigmoid')(encoded2)
encoded2 = Dense(6272, activation='softmax')(encoded2)
encoded3 = Reshape((28, 28, 8))(encoded2)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded3)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(inputs=input_signal, outputs=decoded)
encoder = Model(input_signal, encoded2)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print(autoencoder.summary())

autoencoder.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)

autoencoder.save("conv_autoencoder2.h5")