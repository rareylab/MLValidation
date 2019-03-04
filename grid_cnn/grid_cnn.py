"""
This script implements the convolutional
neural network from:
Ragoza, M., Hochuli, J., Idrobo, E., Sunseri, J., Koes D.R.
Protein–Ligand Scoring with Convolutional Neural Networks.
J.Chem.Inf.Model., 2017, 57 (4), 942-957,
DOI: 10.1021/acs.jcim.6b00740"""
import argparse
from imblearn.over_sampling import RandomOverSampler
import json
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import numpy as np
import sklearn.metrics
import sklearn.utils
from sklearn.externals import joblib
import time


class CNN:

    def __init__(self, input_shape, n_channels=1, callback=None, epochs=1, batch_size=10, nof_cpus=1,
                 max_queue_size=10, conv_dim=3):
        """
        Initialize the learning model.
        :param input_shape: the input shape of a sample (in dense format)
        :param n_channels: number of channels
        :param callback: a Keras callback object
        :param epochs: number of epochs
        :param batch_size: size of a train batch
        :param nof_cpus: number of cpus to use for batch generation
        :param max_queue_size: number of batches in the queue
        """

        self.input_shape = input_shape
        self.n_channels = n_channels
        self.callback = callback
        self.epochs = epochs
        self.batch_size = batch_size
        self.nof_cpus = nof_cpus
        self.max_queue_size = max_queue_size

        self._allowed_params = ['batch_size', 'nof_cpus', 'callback', 'epochs', 'max_queue_size']
        self.channel_order = 'channels_last'  # required by tensorflow backend

        if conv_dim == 3:
            self.model = self._init_model_3D(self.input_shape + (self.n_channels,))
        elif conv_dim == 2:
            self.model = self._init_model_2D(self.input_shape + (self.n_channels,))
        else:
            raise ValueError('Only 2D and 3D convolutions implemented. Aborting')

    def set_params(self, params):
        """
        :param params: dict with parameters allowed are 'batch_size', 'nof_cpus', 'callback', 'epochs', 'max_queue_size'
        """
        self.__dict__.update({k: v for k, v in params.items() if k in self._allowed_params})
        return self

    def fit(self, X_train, y_train, data_generator, batch_gen_type):
        """
        Fits the model with the training data.
        :param X_train: train data
        :param y_train: train labels
        :param data_generator: a instance of Keras Sequence class for providing data
        :param batch_gen_type: a BatchGenerator instance that provides batches
        :return:
        """
        self.model.summary()  # prints a nice to have summary of the model

        self.model.fit_generator(generator=data_generator(X_train, y_train, batch_gen_type,
                                                          batch_size=self.batch_size, dim=self.input_shape,
                                                          n_channels=self.n_channels
                                                          ),
                                 epochs=self.epochs,
                                 callbacks=[self.callback],
                                 max_queue_size=self.max_queue_size,
                                 use_multiprocessing=self.nof_cpus > 1,
                                 workers=self.nof_cpus,
                                 verbose=1  # 2 is one line per epoch
                                 )
        return self

    def _init_model_3D(self, input_shape):
        """
        Initialize the models architecture with 3D convolution.
        :param input_shape: shape of a sample
        :return: the initialized model
        """
        """ Architecture and parameters as in the paper """
        from keras.layers import Conv3D, MaxPooling3D

        model = Sequential()

        model.add(MaxPooling3D(pool_size=(2, 2, 2),
                               strides=(2, 2, 2),
                               padding='valid',
                               data_format=self.channel_order,
                               input_shape=input_shape
                               ))

        model.add(Conv3D(filters=32,
                         kernel_size=3,
                         strides=(1, 1, 1),
                         padding='same',
                         data_format=None,
                         dilation_rate=(1, 1, 1),
                         activation='relu',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         ))

        model.add(Dropout(0.5))

        model.add(MaxPooling3D(pool_size=(2, 2, 2),
                               strides=(2, 2, 2),
                               padding='valid',
                               data_format=self.channel_order))

        model.add(Conv3D(filters=64,
                         kernel_size=3,
                         strides=(1, 1, 1),
                         padding='same',
                         data_format=None,
                         dilation_rate=(1, 1, 1),
                         activation='relu',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None))

        model.add(Dropout(0.5))

        model.add(MaxPooling3D(pool_size=(2, 2, 2),
                               strides=(2, 2, 2),
                               padding='valid',
                               data_format=self.channel_order))

        model.add(Conv3D(filters=128,
                         kernel_size=3,
                         strides=(1, 1, 1),
                         padding='same',
                         data_format=None,
                         dilation_rate=(1, 1, 1),
                         activation='relu',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None))

        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(units=2,
                        activation='softmax',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None))

        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9,
                                                     decay=0.001,
                                                     nesterov=False),
                      metrics=['accuracy'])
        return model

    def _init_model_2D(self, input_shape):
        """
        Initialize models architecture with 2D convolution.
        :param input_shape: shape of a sample
        :return: the initialized model
        """
        from keras.layers import Conv2D, MaxPooling2D

        model = Sequential()

        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='valid',
                               data_format=self.channel_order,
                               input_shape=input_shape
                               ))

        model.add(Conv2D(filters=32,
                         kernel_size=3,
                         strides=(1, 1),
                         padding='same',
                         data_format=None,
                         dilation_rate=(1, 1),
                         activation='relu',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         ))

        model.add(Dropout(0.5))

        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='valid',
                               data_format=self.channel_order))

        model.add(Conv2D(filters=64,
                         kernel_size=3,
                         strides=(1, 1),
                         padding='same',
                         data_format=None,
                         dilation_rate=(1, 1),
                         activation='relu',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None))

        model.add(Dropout(0.5))

        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='valid',
                               data_format=self.channel_order))

        model.add(Conv2D(filters=128,
                         kernel_size=3,
                         strides=(1, 1),
                         padding='same',
                         data_format=None,
                         dilation_rate=(1, 1),
                         activation='relu',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None))

        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(units=2,
                        activation='softmax',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None))

        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9,
                                                     decay=0.001,
                                                     nesterov=False),
                      metrics=['accuracy'])
        return model


class BatchGenerator:
    """
    Implements a simple batch generator.
    """

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

        assert (len(self.samples) == len(self.labels))

    def get_next_batch(self, batch_size, index):
        return self.samples[index * batch_size:(index + 1) * batch_size]

    def get_nof_samples(self):
        return len(self.samples)


class RandomBatchGenerator(BatchGenerator):
    """
    Implements a simple batch generator, which initially shuffles the samples.
    """

    def __init__(self, samples, labels):
        super().__init__(samples, labels)
        self.shuffle()

    def shuffle(self):
        self.samples, self.labels = sklearn.utils.shuffle(self.samples, self.labels)


class OversamplingBatchGenerator(RandomBatchGenerator):
    """
    Batch generator which initially oversamples the minority class to a 1:1 ratio.
    Then samples are shuffled. No explicit stratification of batches, but statistically
    every batch has a 1:1 ratio of classes.
    """

    def __init__(self, samples, labels):
        super(RandomBatchGenerator, self).__init__(samples, labels)
        self._oversample()
        self.shuffle()

    def _oversample(self):
        print('Starting Oversampling')

        nof_actives = np.sum([l[1] for l in self.labels])
        nof_inactives = np.sum([l[0] for l in self.labels])

        if np.ndim(self.samples) == 1:
            self.samples = np.reshape(self.samples, (-1, 1))
        if np.ndim(self.labels) > 1:
            self.labels = np.argmax(self.labels, axis=1)

        ros = RandomOverSampler(random_state=None)  # 42
        self.samples, self.labels = ros.fit_sample(self.samples, self.labels)  # resample
        self.samples = np.reshape(self.samples, -1)  # needs to be reshaped in old shape
        # labels stays the same shape. We only need the labels for the oversampling
        print('finished oversampling')

        print('BEFORE Oversampling: class 0: {} samples, class 1: {} samples'.format(nof_inactives, nof_actives))

        new_nof_actives = np.sum(self.labels)
        new_nof_inactives = len(self.labels) - np.sum(self.labels)

        assert (new_nof_actives == new_nof_inactives)

        print('AFTER Oversampling: class 0: {} samples, class 1: {} samples'.format(new_nof_inactives, new_nof_actives))


class DataGeneratorSparse(keras.utils.Sequence):
    """
    This class provides dense batches from sparse data.
    The sparse representation of the dataset needs far less memory as the dense.
    Although before submitting batches to the network the samples need to be converted to dense format.
    This is the purpose of this class.
    """

    def __init__(self, samples, class_labels, batch_generator, batch_size=10, dim=(48, 48, 48), n_channels=1,
                 dtype=np.bool, shuffle=True):
        """
        Constructs a sparse DataGenerator for supplying batches for training or testing
        :param samples: data samples in sparse format
        :param class_labels: class labels
        :param batch_generator: a batch_generator object, e.g. OversamplingBatchGenerator
        :param batch_size: the size of a single batch
        :param dim: the dimensions of a sample
        :param n_channels: the number of channels for each sample
        :param dtype: data type stored in a samples matrix
        :param shuffle: shuffle the data at epoch end
        """

        assert (batch_size < len(samples))
        assert (len(samples) == len(class_labels))

        self.dim = dim
        self.batch_size = batch_size
        self.samples = np.array(samples)
        self.indices = np.arange(len(self.samples))
        self.labels = np.array(class_labels)
        self.n_channels = n_channels
        self.dtype = dtype
        self.shuffle = shuffle

        self.shape = (batch_size,) + dim + (n_channels,)  # channels are always last

        self.batch_generator = batch_generator(self.indices, self.labels)
        self.nof_samples_per_epoch = self.batch_generator.get_nof_samples()

    def __len__(self):
        """
        Gets number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(self.nof_samples_per_epoch / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch based on the given index.
        :param index: index of the batch
        :return: batch of data
        """
        batch_indices = self.batch_generator.get_next_batch(self.batch_size, index)
        dense_batch = np.zeros(shape=self.shape, dtype=self.dtype)

        for batch_i, sample_indices_list in enumerate(self.samples[batch_indices]):
            for mat_indices in sample_indices_list:
                    dense_batch[(batch_i,) + tuple(mat_indices)] = True

        if len(batch_indices) == self.batch_size:
            return dense_batch, self.labels[batch_indices]
        else:
            return dense_batch[:len(batch_indices)], self.labels[batch_indices]

    def on_epoch_end(self):
        """
        Execute routines at epoch end
        """
        if self.shuffle:
            self.batch_generator.shuffle()

    def get_nof_samples(self):
        """
        Returns actual number of samples provided by this data generator (includes for example oversampled data.
        Depends on batch generator used)
        :return: number of sample per epoch
        """
        return self.nof_samples_per_epoch


class FittingTracker(keras.callbacks.Callback):
    """
    This class is a Keras callback and measures model's performance at the end of each epoch.
    """

    def __init__(self, x_train, y_train, x_test, y_test, ncpus, shape, n_channels, batch_size=50):
        super(keras.callbacks.Callback, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.ncpus = ncpus
        self.shape = shape
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.epoch_counter = 0
        self.measures = {'train': {}, 'test': {}}
        for k in self.measures.keys():
            self.measures[k] = {'auc': [],
                                'loss': [],
                                'acc': [],
                                'tp': [],
                                'tn': [],
                                'fp': [],
                                'fn': []
                                }

    def on_train_end(self, logs={}):

        print('Final Train AUCs {}'.format(self.measures['train']['auc']))
        print('Final Test AUCs {}'.format(self.measures['test']['auc']))
        print('Final Train LOSSs {}'.format(self.measures['train']['loss']))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_counter += 1
        self._calc_metrics(logs)

    def _calc_metrics(self, logs):
        self.measures['train']['loss'].append(logs.get('loss'))

        if self.x_train is not None and self.y_train is not None:
            print('Calculating performance of Train set')

            data_generator = DataGeneratorSparse(self.x_train, self.y_train, BatchGenerator,
                                                 self.batch_size, self.shape, self.n_channels, np.bool,
                                                 shuffle=False)

            preds = self.model.predict_generator(data_generator,
                                                 verbose=1,
                                                 workers=self.ncpus,
                                                 max_queue_size=750,
                                                 use_multiprocessing=True)

            self._set_metric('train', self.y_train, preds)

            print(preds)
            assert (len(preds) == len(self.y_train))

        if self.x_test is not None and self.y_test is not None:
            print('Calculating performance of Test set')

            data_generator = DataGeneratorSparse(self.x_test, self.y_test, BatchGenerator,
                                                 self.batch_size, self.shape, self.n_channels, np.bool,
                                                 shuffle=False)

            preds = self.model.predict_generator(data_generator,
                                                 verbose=1,
                                                 workers=self.ncpus,
                                                 max_queue_size=750,
                                                 use_multiprocessing=True)

            self._set_metric('test', self.y_test, preds)

            print(preds)
            assert(len(preds) == len(self.y_test))

        self._print_metrics()

    def _set_metric(self, data_set, y_true, preds):
        # Some metrics requiere 1d vectors
        y_true_1d = np.argmax(y_true, axis=1)
        assert (np.sum(y_true_1d) == np.sum(y_true[:, 1]))
        preds_1d = np.argmax(preds, axis=1)  # 1d and binary

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true_1d, preds_1d).ravel()

        self.measures[data_set]['auc'].append(sklearn.metrics.roc_auc_score(y_true, preds))
        self.measures[data_set]['acc'].append(sklearn.metrics.accuracy_score(y_true_1d, preds_1d))
        self.measures[data_set]['tp'].append(tp)
        self.measures[data_set]['tn'].append(tn)
        self.measures[data_set]['fp'].append(fp)
        self.measures[data_set]['fn'].append(fn)

    def _print_metrics(self):
        for k, v in self.measures.items():
            for measure, score_list in v.items():
                print('Epoch:{} {} {} {}'.format(self.epoch_counter, k, measure, score_list))


def read_data(paths_list):
    """
    Reads data from one or multiple files
    :param paths_list: list of paths
    :return: input matrix x, label vector y, mapping of targets to rows in the matrix and the
             dense shape of a single sample
    """

    x = []
    y = []
    target_row_mapping = []
    shape_set = set()

    # read descriptor from each path in the list
    for filename in paths_list:
        descriptor_data, grid_shape = joblib.load(filename)

        shape_set.add(grid_shape)
        if len(shape_set) != 1:  # the shape in each file must be equal
            raise ValueError('Grid shapes of the folds are not compatible.')

        for desc, code, y_label in descriptor_data:
            y_ = [0, 1] if y_label == 1 else [1, 0]
            x.extend(desc)
            y.extend([y_ for _ in range(len(desc))])
            target_row_mapping.extend([code] * len(desc))

    assert (len(x) == len(target_row_mapping) == len(y))

    return np.array(x), np.array(y), target_row_mapping, shape_set.pop()


def split(target_row_mapping, folds):
    """
    Generates training and test sets for each cross validation fold specified in folds
    :param target_row_mapping: mapping of samples to protein targets aas list
    :param folds: dict specifying the folds of the cross validation
    :return: a generator yielding train/test indices pairs
    """

    for fold_idx, fold_targets in folds.items():

        # first get the test indices from the fold specification
        test_indices = []
        for target in fold_targets:
            target_low = target.lower()

            if target_low not in set(target_row_mapping):
                raise ValueError('target {} in folds does not exists in input data. '
                                 'Existing targets: {}'.format(target_low, set(target_row_mapping)))

            test_indices.extend([idx for idx, row in enumerate(target_row_mapping) if row == target_low])

        # train indices can be determined by simply using a mask and inversing the test indices
        mask = np.ones_like(target_row_mapping, dtype=bool)
        mask[test_indices] = False
        train_indices = np.argwhere(mask).ravel().tolist()

        # some assertions
        assert(len(train_indices) > 0)
        assert(len(test_indices) > 0)
        assert(len(train_indices) + len(test_indices) == len(target_row_mapping))
        s = set(train_indices)
        s.update(test_indices)
        assert(len(s) == len(target_row_mapping))

        yield train_indices, test_indices


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains and evaluates the grid based 3D (or 2D) convolutional neural'
                                                 ' network.')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True,
                        help='input files')
    parser.add_argument('-f', '--folds', type=str, required=True,
                        help='a json file specifying the folds of cross validation')
    parser.add_argument('-c', '--ncpus', type=int, default=1,
                        help='number of cpus to use')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='number of epochs. Default 1.')
    parser.add_argument('-b', '--batch_size', type=int, default=10,
                        help='size of train batches')
    parser.add_argument('--use2D', action='store_true',
                        help='Use 2D convolution instead of 3D. Input data must have appropriate shape')

    args = parser.parse_args()

    data_generator = DataGeneratorSparse

    print('Starting to read data into memory...')
    s_time = time.time()
    x, y, target_row_mapping, sample_shape = read_data(args.input)
    print('Finished reading data into memory! Took: {} min'.format((time.time() - s_time) / 60.0))

    dims = sample_shape[:-1]
    n_channels = sample_shape[-1]

    with open(args.folds, 'r') as f:
        folds = json.load(f)

    f_idx = 1
    for train, test in split(target_row_mapping, folds):

        print('Fold: {} Train size: {} Test size: {} '.format(f_idx, len(train), len(test)))
        f_idx += 1

        cnn = CNN(dims, n_channels, conv_dim=2 if args.use2D else 3)
        cnn.set_params({'epochs': args.epochs,
                        'batch_size': args.batch_size,
                        'nof_cpus': args.ncpus,
                        'max_queue_size': 750,
                        'callback': FittingTracker(
                            x[train], y[train],
                            x[test], y[test],
                            ncpus=args.ncpus, shape=dims,
                            n_channels=n_channels, batch_size=50)
                        })

        cnn.fit(x[train], y[train], data_generator, OversamplingBatchGenerator)
