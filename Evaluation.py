'''Evaluate trained model'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report

if __name__ == "__main__":
    final_model = tf.keras.models.load_model('weights-modification-training-mod-adam-batchsize-50-99-0.87.hdf5')

    # Load Test Data
    test = np.load('final_test.npy')
    y_test = np.load('final_test_y.npy')

    # Model Evaluation
    accuracy = final_model.evaluate(test, y_test)[1]
    female_accuracy = final_model.evaluate(test[:500], y_test[:500])[1]
    male_accuracy = final_model.evaluate(test[500:], y_test[500:])[1]
    print('Model accuracy is: ', accuracy)
    print('Female Class accuracy is', female_accuracy)
    print('Male Class accuracy is ', male_accuracy)

    # Calculating Class metrics
    predictions = []
    sample = np.ndarray([1,224,224,3])
    for i in range(0,1000):
        sample[0,:,:,:] = test[i]
        if (final_model.predict(sample)[0] < 0.5):
            predictions = predictions + [0]
        else:
            predictions = predictions + [1]

    actual_w = [1]*500
    actual_m = [0]*500
    actual = actual_w + actual_m

    target_names = ['class 0', 'class 1']
    print(classification_report(actual, predictions, target_names=target_names))
