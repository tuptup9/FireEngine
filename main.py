import tensorflow as tf  # Used as the main framework
from tensorflow.keras.applications import MobileNetV2  # Main model
import datetime  # Used for tensorboard logs
from data import *  # This file contains all dataset handling functions
from tensorflow.keras import callbacks  # Used for checkpoints and early stopping

### Project overview
# The code presented here is for a hyperheuristics method based on a convolutional network.
# Its purpose is to select from a given set of heuristics (set to 4 currently) based
# on data generated from checking which heuristic is optimal in any given graph step.

# Some considerations:
# --The main idea is to treat each graph like an image. Since we resize the input
#  using normal image resizing techniques, you will see float values in the matrices
#  (in a graph, one would not expect them).
# --This particular part of the code is for training the network. For usage after
#  training, one should consult the FFP.py file.


### Parameters
dims = (516, 516)  # Size of graph (will resize to this if necessary)
batch = 3
seed = 123  # Seed for the random operations
output_size = 2  # Heuristics to be used
epochs = 50  # How many training epochs (max)? Note that we set early-stopping to true.

### Model
# We use MobileNetV2 as a good, general-use convolutional network for classification
# Being lightweight is a plus for us, due to limited computational power.
classifier = MobileNetV2(
    input_shape=dims + (1,),  # Set for a greyscale image of the desired size
    include_top=True,  # We include the classifier
    weights=None,  # For loading, we set this to the saved weights
    classes=output_size,  # One for each heuristic
    classifier_activation='softmax',  # Activation function for the classifier
)

classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy',
                   metrics='accuracy')

classifier.summary()

### Data set
# Our goal here: To have the model train automatically from a given set of graphs

# load_all_data(dims) # Run to generate save file from dataset
data = np.load('graphs_matrices.npy')
gt = np.load('graph_gt.npy')
gt = tf.keras.utils.to_categorical(gt, num_classes=output_size)

### Training
# Callbacks
model_checkpoint = callbacks.ModelCheckpoint(
    'fireEngine.hdf5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_freq="epoch"
)
log_dir = "F:\\Biggums Filus\\Teravolt\\logs\\fit\\fe\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# In order, the callbacks are:
# - Save the best model every epoch
# - Save logs for tensorboard
callbacks = [model_checkpoint, tensorboard_callback]

# Begin training
classifier.fit(
    x=data,
    y=gt,
    batch_size=batch,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_split=0.05,
    shuffle=True,
)

