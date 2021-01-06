"""Define pre-trained model (Frozen) and new top layers.""" 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Input, BatchNormalization, Dropout


def gen_base_model():
    model1 = tf.keras.Sequential()

    img = model1.add(Input(shape=(224, 224,3)))
    Conv1_1 = model1.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))  # Conv1_1
    Conv1_2 = model1.add(Conv2D(filters=64,kernel_size=(3, 3),padding="same", activation="relu"))  # Conv1_2
    mp_1 = model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))  # MP_1

    Conv2_1 = model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))  # Conv2_1
    Conv2_2 = model1.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 2_2
    mp_2 = model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))  # MP_2

    Conv3_1 = model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 3_1
    Conv3_2 = model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 3_2
    Conv3_3 = model1.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 3_3
    mp3 = model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))  # MP_3

    Conv4_1 = model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 4_1
    Conv4_2 = model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 4_2
    Conv4_3 = model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 4_3
    mp4 = model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))  # MP_4

    Conv5_1 = model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 5_1
    Conv5_2 = model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 5_2
    Conv5_3 = model1.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))  # Conv 5_3
    mp5 = model1.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))  # MP_5
    
    model1.add(Flatten())
    fc_6 = model1.add(Dense(units=4096,activation="relu"))  # FC-6

    model1.trainable = False
    
    return model1

def new_top_model():
    model = tf.keras.models.Sequential()
    fc_7 = model.add(Dense(units=256,activation="relu"))  #  FC-7
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    fc_8 = model.add(Dense(units=256,activation="relu"))  #  FC-8
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    fc_9 = model.add(Dense(units=1, activation="sigmoid"))  # FC-9
    
    return(model)

