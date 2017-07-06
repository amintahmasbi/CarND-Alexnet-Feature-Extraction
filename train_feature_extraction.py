import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from alexnet import AlexNet

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43

# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']

X_train =X_train.astype('float32')

#X_train -= np.mean(X_train,axis=(1,2,3)).reshape((X_train.shape[0],1,1,1))
# TODO: Split data into training and validation sets.
validation_split_portion = 0.2
X_train, y_train = shuffle(X_train, y_train)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_split_portion)
# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x,(227,227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

fc8W = tf.Variable(tf.truncated_normal(shape=shape,stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
EPOCHS = 30
BATCH_SIZE = 128 

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
# TODO: Train and evaluate the feature extraction model.
#model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, 'AlexNet_GSTRB_network')
    
