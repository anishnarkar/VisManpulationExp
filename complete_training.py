import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from branched_model import AgeNet

def lr_scheduler(epoch, learning_rate):
    initial_learning_rate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    learning_rate = initial_learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return learning_rate

def train_model(compound_model, train, y_train, validation, y_validation, epochs, batch_size, optimizer='sgd'):

    # Early Stopping

    if epochs > 50:
        patience = .20*epochs
    else:
        patience = .30*epochs

    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.1, baseline = 0.87, patience = patience)

    # learning schedule callback
    learning_rate = LearningRateScheduler(lr_scheduler)

    # Checkpointing
    filepath="weights-modification-" + optimizer.lower() + "-batchsize-" + str(batch_size) + "-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Tensorboard
    tensorboard_callback = TensorBoard(log_dir="./logs")

    # Optimizer
    
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        callback_list = [early_stop, checkpoint, tensorboard_callback]
        
    elif optimizer.lower() == 'sgd':
        momentum = 0.8
        opt = tf.keras.optimizers.SGD(lr=0.0, momentum=momentum)
        callback_list = [early_stop, learning_rate, checkpoint, tensorboard_callback]

    # Compile model
    compound_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Train model
    compound_model.fit(train, y_train, validation_data=(validation, y_validation), epochs=epochs, batch_size=batch_size, callbacks=callback_list)
    
    return compound_model

if __name__ == "__main__":

    # Load data

    # Training
    train = np.load('final_train.npy')
    y_train = np.load('final_train_y.npy')

    # Validation
    validation = np.load('final_val.npy')
    y_validation = np.load('final_val_y.npy')

    # Test
    test = np.load('final_test.npy')
    y_test = np.load('final_test_y.npy')

    # Define Models and load pre-trained weights

    base_model = generate_models.gen_base_model()
    base_model.load_weights('completemodel.hdf5')

    top_layers = generate_models.new_top_model()
    top_layers.build(input_shape=[None,4096])
    
    

    # Define a combined model

    compound_model = tf.keras.Sequential([
      base_model,
      top_layers
    ])

    # Compile and train model

    compound_model = train_model(compound_model, train, y_train,
                                 validation, y_validation, epochs = 100,
                                 batch_size = 40, optimizer='adam')

    # Model Evaluation

    accuracy = compound_model.evaluate(test, y_test)[1]
    print('Model accuracy is: ', accuracy)
