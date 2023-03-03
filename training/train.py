
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Layer,
                                     MaxPooling2D)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model

from config import get_settings

settings = get_settings()

# If running on GPU - set limitaiton for GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Image directories
ANC_PATH = settings.ANC_PATH
POS_PATH = settings.POS_PATH
NEG_PATH = settings.NEG_PATH
EPOCHS = settings.EPOCHS

# Get image addrecess
anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg')
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg')
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg')

# Image preprocessing 
def preprocss(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0

    return img 

# Create labelled dataset
positives = tf.data.Dataset.zip((
   anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))
   ))
negatives = tf.data.Dataset.zip((
    anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))
    ))
data = positives.concatenate(negatives)

def preprocss_twin(input_img, validation_img, label):

    return (preprocss(input_img), preprocss(validation_img), label)

# Build dataloader pipeline
data = data.map(preprocss_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Train-test split
train_data = data.take(round(len(data)*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Test data

test_data = data.skip(round(len(data)*0.7))
test_data = test_data.take(round(len(data)*0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

     # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    # Final block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')

model = make_embedding()

# Build Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
  
    def call(self, input_embedding, validation_emdedding):
        return tf.math.abs(input_embedding - validation_emdedding)
  
# Make Siamese Network
def make_siamese_model():
    # Anchor image
    input_image = Input(name='input_img', shape=(100,100,3))

    # validtion image
    validation_image = Input(name='validation_img', shape=(100,100,3))

    # Adding embedding to distance calculation
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(model(input_image), model(validation_image))

    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()

# Loss function:
binary_cross_loss = tf.losses.BinaryFocalCrossentropy()

# Optimizer 
opt = tf.keras.optimizers.Adam(1e-4)

# Establishing Checkpoints: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix= os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Training step function:
@tf.function
def train_step(batch):

    with tf.GradientTape() as tape:

         #get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass 
        yhat = siamese_model(X, training=True)
        # Calculating loss
        loss = binary_cross_loss(y, yhat)

    # Calculating gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
  
    return loss

# Building Training loop 
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch,EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

    for idx, batch in enumerate(data):
        train_step(batch)
        progbar.update(idx+1)

    # save checkpoints every 10 epochs
    if epoch % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)


# Training the model
EPOCHS = 50
train(train_data, EPOCHS)
print("Training Completed")

# save weights
siamese_model.save('siamesemodel1.h5')
print("Model saved")


## Model Evaluation

# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])

y_hat = [1 if prediction > 0.5 else 0 for prediction in y_hat]

# calculating recall
metric = Recall()
metric.update_state(y_true, y_hat)
recall = metric.result().numpy()
print(f"Recall is: {recall}")

# calculating precision

metric = Precision()
metric.update_state(y_true, y_hat)
precision = metric.result().numpy()
print(f"Precision is: {precision}")

