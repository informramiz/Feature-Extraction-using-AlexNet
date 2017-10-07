import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

#define common constants
nb_classes = 43
epochs = 10
batch_size = 128

# Load traffic signs data.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

# Split data into training and 33% validation sets.
X_train, X_validation, y_train, y_validation = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

# Define placeholders and resize operation.
#our image size is 32x32x3
features = tf.Variabale(tf.float32, (None, 32, 32, 3))
labels = tf.Variable(tf.int32, None)
#AlexNet expects images to be 277x277x3
resized_features = tf.image.resize_images(features, (227, 227))

# pass features placeholder as first argument to `AlexNet`.
#pass feature_extract=True means that we want last layer: fc7
#for feature extraction
fc7 = AlexNet(resized_features, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
fc7_features_count = fc7.get_shape().as_list[1]
fc8W = tf.Variable(tf.truncated_normal((None, fc7_features_count, nb_classes)))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.xw_plus_b(fc7, fc8W, fc8b)

# Define loss, training, accuracy operations.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits)
loss_op = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
optimze_op = optimizer.minimize(loss, var_list=[fc8W, fc8W])

predictions_match = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
accuracy_op = tf.reduce_mean(tf.cast(predictions_match, tf.float32))

# Train and evaluate the feature extraction model.
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    t0 = time.time()
    for offset in range(0, X_train.shape[0], batch_size):
        end = offset + batch_size
        session.run([optimze_op], feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

    val_loss, val_acc = evaluate(session, X_validation, y_validation)
    print("Epoch: ", i+1)
    print("Time: %.3f seconds" % (time.time() - t0))
    print("Validation Loss =", val_loss)
    print("Validation Accuracy =", val_acc)
    print()

#Define method to evaluate validation set
def evaluate(session, X, y):
    total_loss = 0
    total_acc = 0

    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = session.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels:y_batch})
        total_loss += (loss * X.shape[0])
        total_acc += (acc * X.shape[0])

    return total_loss / X.shape[0], total_acc / X.shape[0]
