# general imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy import special

# tensorflow imports
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1,l2

# tensorflow-privacy
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType

def load_MNIST():
    """Loads MNIST-Dataset and preprocesses to combine training and test data."""

    # load the existing MNIST dataset that comes in form of traing + test data and labels
    train, test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    # scale the images from color values 0-255 to numbers from 0-1 to help the training process
    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    # cifar10 labels come one-hot encoded, there
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels
def load_cifar10():
    """Loads Cifar10-Dataset and preprocesses to combine training and test data."""

    # load the existing CIFAR10 dataset that comes in form of traing + test data and labels
    train, test = tf.keras.datasets.cifar10.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    # scale the images from color values 0-255 to numbers from 0-1 to help the training process
    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    # cifar10 labels come one-hot encoded, there
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels
def load_fashion_mnist():
    """Loads fashion_mnist Dataset and preprocesses to combine training and test data."""

    # load the existing fashion_mnist dataset that comes in form of traing + test data and labels
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    # scale the images from color values 0-255 to numbers from 0-1 to help the training process
    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    # cifar10 labels come one-hot encoded, there
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels
def load_datasets(dataset):
    """Loads fashion_mnist Dataset and preprocesses to combine training and test data."""

    # load the existing fashion_mnist dataset that comes in form of traing + test data and labels
    if dataset == 'MNIST':                      # (70000,28,28)
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
        train_data = train_data.reshape(-1, 28, 28, 1)
        test_data  =  test_data.reshape(-1, 28, 28, 1)
    elif dataset == 'cifar10':                  # (60000,32,32,3)
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'Fashion_mnist':            # (70000,28,28)
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_data = train_data.reshape(-1, 28, 28, 1)
        test_data  =  test_data.reshape(-1, 28, 28, 1)
    else:
        print('Dataset Error !!!')
        return

    # scale the images from color values 0-255 to numbers from 0-1 to help the training process
    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    # cifar10 labels come one-hot encoded, there
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    return train_data, train_labels, test_data, test_labels
def make_simple_model():
    """ Define a Keras model without much of regularization. Such a model is prone to overfitting"""
    shape = (32, 32, 3)
    i = Input(shape=shape)
    x = Conv2D(32, (3, 3), activation='relu')(i)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # if we don't specify an activation for the last layer, we can have the logits
    x = Dense(10)(x)
    model = Model(i, x)
    return model
def make_simple_model(dataset):
    """ Define a Keras model without much of regularization. Such a model is prone to overfitting"""
    if dataset == 'MNIST':                      # (70000,28,28)
        shape = (28, 28, 1)
    elif dataset == 'cifar10':                  # (60000,32,32,3)
        shape = (32, 32, 3)
    elif dataset == 'Fashion_mnist':            # (70000,28,28)
        shape = (28, 28, 1)
    else:
        print('Dataset Error !!!')
        return

    i = Input(shape=shape)

    x = Conv2D(32, (3, 3), activation='relu')(i)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu',kernel_regularizer=l2(0.0003))(x)
    # if we don't specify an activation for the last layer, we can have the logits
    x = Dense(10,kernel_regularizer=l2(0.0003))(x)
    model = Model(i, x)
    return model
# train_data, train_labels, test_data, test_labels = load_cifar10()
# train_data, train_labels, test_data, test_labels = load_MNIST()
# train_data, train_labels, test_data, test_labels = load_fashion_mnist()
# [ 'cifar10', 'MNIST','Fashion_mnist']

for dataset in [ 'cifar10']:
    # dataset = 'Fashion_mnist' # MNIST, cifar10, Fashion_mnist
    train_data, train_labels, test_data, test_labels = load_datasets(dataset)
    print(np.shape(train_data),np.shape(train_labels),np.shape(test_data),np.shape(test_labels))
    # show some train data
    num_row ,num_col = 2 , 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(10):
        ax = axes[i//num_col, i%num_col]
        ax.set_axis_off()
        ax.imshow(train_data[i])

    plt.tight_layout()
    plt.show()

    # make the neural network model with the function specified above.
    # one model is supposed to train for 10, one for 50 epochs
    model = make_simple_model(dataset)
    model.summary()

    # specify parameters
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # train the model
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),batch_size=128,epochs=30)

    # plot accuracy for the first model

    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.ylim(0,1)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Simple Mode trained by '+ dataset)
    plt.show()

    # since we have not specified an activation function on the last layer
    # calling the predict function returns the logits
    print('Predict on train...')
    logits_train = model.predict(train_data)
    print('Predict on test...')
    logits_test = model.predict(test_data)

    print('Apply softmax to get probabilities from logits...')
    prob_train = special.softmax(logits_train, axis=1)
    prob_test = special.softmax(logits_test, axis=1)

    print('Compute losses...')
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant

    y_train_onehot = to_categorical(train_labels)
    y_test_onehot = to_categorical(test_labels)

    loss_train = cce(constant(y_train_onehot), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test_onehot), constant(prob_test), from_logits=False).numpy()

    # define what variables our attacker should have access to
    attack_input = AttackInputData( logits_train = logits_train,logits_test = logits_test,loss_train = loss_train, loss_test = loss_test, labels_train = train_labels,labels_test = test_labels)

    # how should the data be sliced
    slicing_spec = SlicingSpec(entire_dataset = True,by_class = True, by_percentiles = False,by_classification_correctness = True)

    # define the type of attacker model that we want to use
    attack_types = [ AttackType.THRESHOLD_ATTACK, AttackType.LOGISTIC_REGRESSION ]

    # run the attack
    attacks_result = mia.run_attacks(attack_input=attack_input, slicing_spec=slicing_spec, attack_types=attack_types)

    # summary by data slice (the best performing attacks per slice are presented)
    print(attacks_result.summary(by_slices=True))

    # plot the curve, we see that the attacker is much better than random guessing
    import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting
    # print(plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve))
    figure = plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve)
    figure.suptitle(dataset)
    figure.show()
    # plt.title(dataset)
    # plt.show()