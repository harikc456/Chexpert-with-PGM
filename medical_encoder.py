import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
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

images = images.reshape(-1,img_size)

images = images/255.0

print(tf.config.experimental.list_physical_devices('GPU'))
 
epochs = 100
 
X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size = 0.3,random_state = 10)

encoding_dim = 64
batch_size = 128

images = images.reshape(-1,img_size)

input_img = Input(shape=(img_size))
encoded_1 = Dense(512, activation='relu')(input_img)
encoded_2 = Dense(256, activation='relu')(encoded_1)
encoded_3 = Dense(128, activation='relu')(encoded_2)
encoded_4 = Dense(encoding_dim, activation='relu')(encoded_3)
decoded_1 = Dense(128, activation='sigmoid')(encoded_4)
decoded_2 = Dense(256, activation='sigmoid')(decoded_1)
decoded_3 = Dense(512, activation='sigmoid')(decoded_2)
decoded_4 = Dense(img_size, activation='sigmoid')(decoded_3)
autoencoder = Model(input_img, decoded_4)

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

autoencoder.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')

autoencoder.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)

autoencoder.save("autoencoder2.h5")