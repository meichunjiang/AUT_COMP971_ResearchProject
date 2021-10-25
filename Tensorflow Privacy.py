# general imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from scipy import special

# tensorflow imports
from keras.models import Sequential
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
    model.summary()

    return model

def load_datasets(dataset):
    """Loads fashion_mnist Dataset and preprocesses to combine training and test data."""

    # load the existing fashion_mnist dataset that comes in form of traing + test data and labels
    if dataset == 'MNIST':                      # (70000,28,28)
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
        train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
        test_data  =  test_data.reshape(test_data.shape[0], 28, 28, 1)
    elif dataset == 'cifar10':                  # (60000,32,32,3)
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'Fashion_mnist':            # (70000,28,28)
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
        test_data  =  test_data.reshape(-test_data.shape[0], 28, 28, 1)
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
def make_VGG16_model(dataset):
    # 输入层
    # load the existing fashion_mnist dataset that comes in form of traing + test data and labels
    if dataset == 'MNIST':                      # (70000,28,28)
        inputs = Input(shape=(28, 28, 1))
    elif dataset == 'cifar10':                  # (60000,32,32,3)
        inputs = Input(shape=(32, 32, 3))
    elif dataset == 'Fashion_mnist':            # (70000,28,28)
        inputs = Input(shape=(28, 28, 1))
    else:
        print('Dataset Error !!!')
        return

    # 卷积层和最大池化层
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=2,padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    pool2 = MaxPooling2D(pool_size=2,padding='same')(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
    conv7 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv6)
    pool3 = MaxPooling2D(pool_size=2,padding='same')(conv7)

    conv8 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv9 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv8)
    conv10 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv9)
    pool4 = MaxPooling2D(pool_size=2,padding='same')(conv10)

    conv11 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool4)
    conv12 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv11)
    conv13 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv12)
    pool5 = MaxPooling2D(pool_size=2,padding='same')(conv13)

    # 扁平层
    flat = Flatten()(pool5)

    # 全联接层
    fc1 = Dense(4096, activation='relu')(flat)
    fc2 = Dense(4096, activation='relu')(fc1)

    # 输出层
    outputs = Dense(1000, activation='softmax')(fc2)

    my_VGG16_model = Model(inputs=inputs, outputs=outputs)
    my_VGG16_model.summary()

    return my_VGG16_model
def make_AlexNet_model(dataset):

    if dataset == 'MNIST':                      # (70000,28,28)
        inputs = (28, 28, 1)
    elif dataset == 'cifar10':                  # (60000,32,32,3)
        inputs = (32, 32, 3)
    elif dataset == 'Fashion_mnist':            # (70000,28,28)
        inputs = (28, 28, 1)
    else:
        print('Dataset Error !!!')
        return

    model = Sequential()
    # conv1
    model.add(Conv2D(96, (11, 11), strides=(1, 1), padding='same', activation='relu', input_shape=inputs))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # conv2
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # conv3
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # conv4
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    # conv5
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # fc5
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.8))
    # fc6
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.8))
    # fc7
    model.add(Dense(10, activation='softmax'))

    model.summary()
    return model

def make_simple_model(dataset,regularization = 'No', coefficient = 0.0003):
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
    if regularization == 'Yes':
        x = Dense(128, activation='relu',kernel_regularizer=l2(coefficient))(x)
        # if we don't specify an activation for the last layer, we can have the logits
        x = Dense(10,kernel_regularizer=l2(coefficient))(x)
    else:
        x = Dense(128, activation='relu')(x)
        # if we don't specify an activation for the last layer, we can have the logits
        x = Dense(10)(x)
    model = Model(i, x)
    model.summary()
    return model


# case 1
def case1 ():
    for dataset in [ 'Fashion_mnist']:                # [ 'cifar10', 'MNIST','Fashion_mnist']
        for coefficient in[5,1,0.1,0.01,0.001,0.0001]:#0.1,0.01,0.001,0.0001
            # load dataset
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

            epochs = 50
            # regularization = 'Yes'
            regularization = 'No'
            # coefficient = 0.0003

            # title = 'SimpleModel trained on '+dataset+ ' for '+str(epochs)+' times.'+'(Regularization = '+regularization+')'
            title = 'SimpleModel_ '+dataset+ '_'+str(epochs)+' times_'+'(Regularization = '+regularization+')'+str(coefficient)
            # make the neural network model with the function specified above.
            # one model is supposed to train for 10, one for 50 epochs
            model = make_simple_model(dataset,regularization,coefficient)
            # model.summary()

            # specify parameters
            optimizer = tf.keras.optimizers.Adam()
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            # compile the model
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            # train the model
            history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),batch_size=128,epochs=epochs)

            # plot accuracy for the first model
            # plt.plot(history.history['accuracy'], label='acc')
            # plt.plot(history.history['val_accuracy'], label='val_acc')
            # plt.legend()
            # plt.ylim(0,1)
            # plt.xlabel('epochs')
            # plt.ylabel('accuracy')
            # plt.title(title)
            # plt.savefig('./'+title+'_Model.jpg')
            # plt.show()

            # plot Loss for the first model
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.legend()
            plt.ylim(0,2)
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.title(title)
            plt.savefig('./'+title+'_Model.jpg')
            plt.show()

            logfile_name = title+'.txt'
            if os.path.exists(logfile_name):
                os.remove(logfile_name)

            logfile = open(logfile_name,'w')
            print(history.history['loss'],file= logfile)
            print(history.history['accuracy'],file= logfile)
            print(history.history['val_loss'],file= logfile)
            print(history.history['val_accuracy'],file= logfile)



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
            print(attacks_result.summary(by_slices=True),file= logfile)


            roc_Yes = attacks_result.get_result_with_max_auc().roc_curve


            # plot the curve, we see that the attacker is much better than random guessing
            import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting
            print(plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve))
            figure = plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve)
            figure.suptitle(title)
            figure.show()
            figure.savefig('./'+title+'_MIA.jpg')

            logfile.close()


# case 2
def case2_model_training():
    epochs = 30
    for dataset in ['MNIST', 'cifar10', 'Fashion_mnist']:  # MNIST, cifar10, Fashion_mnist
        # load dataset
        train_data, train_labels, test_data, test_labels = load_datasets(dataset)
        print(np.shape(train_data), np.shape(train_labels), np.shape(test_data), np.shape(test_labels))

        # show some train data
        # num_row ,num_col = 2 , 5
        # fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        # for i in range(10):
        #     ax = axes[i//num_col, i%num_col]
        #     ax.set_axis_off()
        #     ax.imshow(train_data[i])
        # plt.tight_layout()
        # plt.show()

        model_simple = make_simple_model(dataset)
        model_alexnet = make_AlexNet_model(dataset)
        model_vgg16 = make_VGG16_model(dataset)

        # specify parameters
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # compile the model
        model_simple.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        model_alexnet.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])
        model_vgg16.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # train the model
        history_simple = model_simple.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=128, epochs=epochs)
        history_alexnet = model_alexnet.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=128, epochs=epochs)
        history_vgg16 = model_vgg16.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=128, epochs=epochs)
        print(history_simple.history)
        print(history_alexnet.history)
        print(history_vgg16.history)

        # plot Loss for the first model
        plt.plot(history_simple.history['loss'], label='Simple train loss')
        plt.plot(history_simple.history['val_loss'], label='Simple val_loss')

        plt.plot(history_alexnet.history['loss'], label='AlexNet train loss')
        plt.plot(history_alexnet.history['val_loss'], label='AlexNet val_loss')

        plt.plot(history_vgg16.history['loss'], label='Vgg16 train loss')
        plt.plot(history_vgg16.history['val_loss'], label='Vgg16val_loss')

        # save history - simple model
        model_simple.save('SaveModel_simple_' + str(dataset))
        model_simple.save_weights(str(dataset) + '_model_simple.hdf5')
        with open(str(dataset) + '_model_simple.txt', 'wb') as file_simple:
            pickle.dump(history_simple.history, file_simple)
        file_simple.close()
        print(history_simple.history)

        # save history - alexnet model
        model_simple.save('SaveModel_AlexNet_' + str(dataset))
        model_alexnet.save_weights(str(dataset) + '_model_AlexNet.hdf5')
        with open(str(dataset) + '_model_AlexNet.txt', 'wb') as file_AlexNet:
            pickle.dump(history_alexnet.history, file_AlexNet)
        file_AlexNet.close()
        print(model_alexnet.history)

        # save history - Vgg16 model
        model_simple.save('SaveModel_Vgg16_' + str(dataset))
        model_vgg16.save_weights(str(dataset) + '_model_Vgg16.hdf5')
        with open(str(dataset) + '_model_Vgg16.txt', 'wb') as file_Vgg16:
            pickle.dump(history_vgg16.history, file_Vgg16)
        file_Vgg16.close()
        print(model_vgg16.history)

        # reload history
        # model_simple.load_weights(str(dataset)+'_model_simple.hdf5')
        # with open(str(dataset)+'_model_simple.txt', 'rb') as f:
        #     reconstructed_history = pickle.load(f)
        #
        # print("reload history:")
        # print(reconstructed_history)

        plt.legend()
        plt.ylim(0, 2)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training Dataset : ' + str(dataset))
        plt.savefig('./' + str(dataset) + 'Model_compare.jpg')
        plt.show()
def case2_model_MIA():
    # reload model
    for dataset in ['MNIST', 'cifar10', 'Fashion_mnist']:  #
            # since we have not specified an activation function on the last layer
            # calling the predict function returns the logits

            # load dataset
            train_data, train_labels, test_data, test_labels = load_datasets(dataset)
            print(np.shape(train_data), np.shape(train_labels), np.shape(test_data), np.shape(test_labels))

            for modelname in ['simple','AlexNet','Vgg16']:
                reconstruct_model = './case2/'+'SaveModel_'+modelname+'_'+dataset
                print(reconstruct_model)
                # load model
                model = tf.keras.models.load_model(reconstruct_model)

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
                # print(attacks_result.summary(by_slices=True),file= logfile)


                roc_Yes = attacks_result.get_result_with_max_auc().roc_curve


                # plot the curve, we see that the attacker is much better than random guessing
                import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting
                print(plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve))
                figure = plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve)
                title = modelname + '_'+ dataset
                figure.suptitle(title)
                figure.show()
                figure.savefig('./'+title+'_MIA.jpg')

def Case2_History_reload():
    path = './case2_Ver2.0/'
    for dataset in ['MNIST', 'cifar10', 'Fashion_mnist']:

        # file = open(path + str(dataset) + '_model_Vgg16.txt', 'r+')
        # print(pickle.load(file))

        # reload history

        with  open(path + str(dataset) + '_model_Vgg16.txt', 'r+') as f:
            reconstructed_history = pickle.load(f)

        print("reload history:")
        print(reconstructed_history)

        # # reload Model simple
        # with open(path + str(dataset) + '_model_simple.pickle', 'rb') as file_simple:
        #     print(path + str(dataset) + '_model_simple.pickle')
        #     print(pickle.load(file_simple))
        #     # simplemodel_history = file_simple.read()


        # # reload Model Vgg16
        # with open(path + str(dataset) + '_model_Vgg16.pickle', 'rb') as file_Vgg16:
        #     Vgg16model_history = file_Vgg16.read()
        #
        # # reload Model AlexNet
        # with open(path + str(dataset) + '_model_AlexNet.pickle', 'rb') as file_AlexNet:
        #     AlexNet_history = file_AlexNet.read()

        # # plot Loss for the first model
        # plt.plot(simplemodel_history.history['loss'], label='Simple train loss')
        # plt.plot(simplemodel_history.history['val_loss'], label='Simple val_loss')
        #
        # plt.plot(Vgg16model_history.history['loss'], label='AlexNet train loss')
        # plt.plot(Vgg16model_history.history['val_loss'], label='AlexNet val_loss')
        #
        # plt.plot(AlexNet_history.history['loss'], label='Vgg16 train loss')
        # plt.plot(AlexNet_history.history['val_loss'], label='Vgg16val_loss')
        #
        # plt.legend()
        # plt.ylim(0,2)
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.title('Training Dataset : '+str(dataset))
        # plt.savefig(str(path)+str(dataset) + '_Model_compare.jpg')
        # plt.show()

Case2_History_reload()