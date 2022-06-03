import numpy as np 
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from contextlib import redirect_stdout
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

print(tf.config.list_physical_devices())

f = open("log.txt", "a")
train_df = pd.read_csv('train/train.csv')
trainImagesPath = ('train_images')
testImagesPath = ('test_images')

def load_image(path, pet_id):
    image = cv2.imread(f'{path}/{pet_id}-1.jpg')
    return image

def load_images(path, pet_id): 
    pictures=glob.glob(f'{path}/{pet_id}-*.jpg')
    return pictures,len(pictures)

trainCNN_df=train_df.copy()

trainCNN_df=trainCNN_df[trainCNN_df.PhotoAmt!=0]

image_df=[]
for index, row in trainCNN_df.iterrows(): 
    imageList,imageCount = load_images(trainImagesPath, row['PetID'])
    if imageCount==0:
        continue
    speed=row['AdoptionSpeed']
    for image in imageList:
        image_df.append([image,speed])
                                                             
f.write(f' we found: {len(image_df)} amount of pictures in the database\n\n\n')
image_df = pd.DataFrame(image_df, columns=['ImageURL','Speed'])

image_df['Speed']=image_df['Speed'].astype(str)

f.write("dataframe:\n{}".format(image_df))

f.write("Speed value counts:{}\n\n\n".format(image_df['Speed'].value_counts()))

pics=image_df['ImageURL']
label=image_df['Speed']
val_split = 0.25
X_train, X_val, y_train, y_val = train_test_split(pics, label, test_size=val_split,stratify=label)

train_CNN = np.concatenate((X_train, y_train))
val_CNN = np.concatenate((X_val, y_val))

train_CNN=pd.DataFrame(list(zip(X_train, y_train)),
              columns=['ImageURL','Speed'])

val_CNN=pd.DataFrame(list(zip(X_val,y_val)),
              columns=['ImageURL','Speed'])

f.write(f' the length of train set is: {len(train_CNN)},the length of validation set is: {len(val_CNN)}\n\n\n')

datagen=ImageDataGenerator(rescale=1./255)

train_generator=datagen.flow_from_dataframe(dataframe=train_CNN, x_col="ImageURL", y_col="Speed", class_mode="categorical", target_size=(112,112), batch_size=32,subset="training",shuffle=True,color_mode="rgb")
valid_generator=datagen.flow_from_dataframe(dataframe=val_CNN, x_col="ImageURL", y_col="Speed", class_mode="categorical", target_size=(112,112), batch_size=32,subset="validation",shuffle=True,color_mode="rgb")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(112,112,3))

base_model.trainable = False 


model = models.Sequential([
    base_model,
    layers.Flatten(), 
    layers.Dense(512, activation='sigmoid'),
    layers.Dense(5, activation='softmax') 
])


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=1000,
        decay_rate=0.96)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

model.compile(
        loss='binary_crossentropy',
        optimizer='RMSprop',
        metrics=['accuracy']
    )

with open('log.txt', 'a') as file: 
    with redirect_stdout(file):
        model.summary()

f.write('\n\n\n')

epochs=100
history=model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=32)
