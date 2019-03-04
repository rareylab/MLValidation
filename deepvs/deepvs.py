"""
This script implements the convolutional
neural network from:
Pereira, J.C., Caffarena, E.R., dos Santos, C.N.
Boosting Docking-Based Virtual Screening with Deep Learning.
J.Chem.Inf.Model., 2016, 56, 5495-2506
"""
import pandas as pd
import math
import tensorflow as tf
import sklearn.metrics
import json
import numpy as np
import operator
import argparse


def my_auc(y_true, y_score, reversed_order=False):
    """
    Calculates ROC AUC score as described in the paper of Pereira et al. 2016
    :param y_true: the ground truth
    :param y_score: the predicted score
    :param reversed_order: whether to reverse order
    :return:
    """
    concat = np.concatenate((np.reshape(y_score, (len(y_score), 1)), np.reshape(y_true, (len(y_true), 1)),), axis=1)
    # we need an descending ordering from best scores to worst scores
    concat_s = sorted(concat, key=operator.itemgetter(0), reverse=not reversed_order)
    n_actives = sum(y_true)
    n_decoys = len(y_true) - n_actives

    auc = 0.0

    decoys_seen = 0
    for _, label in concat_s:
        if not label:
            decoys_seen += 1
        else:
            auc += decoys_seen / n_decoys

    return 1 - auc / n_actives


def my_ef(y_true, y_score, fac, reversed_order=False):
    """
    Calculates the enrichment factor as described in Pereira et al. 2016
    :param y_true: the ground truth
    :param y_score: the predicted score
    :param fac: enrichment for the top fac ranked samples
    :param reversed_order: whether to reverse the ordering
    :return:
    """
    concat = np.concatenate((np.reshape(y_score, (len(y_score), 1)), np.reshape(y_true, (len(y_true), 1)),), axis=1)
    # we need an descending ordering from best scores to worst scores
    concat_s = sorted(concat, key=operator.itemgetter(0), reverse=not reversed_order)
    n_actives = sum(y_true)
    n_compounds = len(y_true)

    max_index = int(fac * n_compounds)

    actives_at_fac = 0

    for i in range(max_index):
        actives_at_fac += 1 if concat_s[i][1] else 0

    return actives_at_fac / max_index * n_compounds / n_actives


class DeepVSModel:
    """
    Represents the DeepVS deep learning model.
    """

    def __init__(self, d_atom, d_dist, d_chrg, cf, h, atom_vocab_size, dist_vocab_size, chrg_vocab_size,
                 max_number_of_atoms, sequence_length, learning_rate):
        """
        Initialize the deep learning model.

        d_atom: size of the embeding for the atom types
        d_dist: size of the embeding for the distances
        d_chrg: size of the embeding for the charges
        cf: number of convolutional filters
        h: number of hidden layers
        atom_vocab_size: size of the vocabulary of atom types == number of different atom types == |A| + 1
        dist_vocab_size: size of the vocabulary of distances  == number of different distance   == |D| + 1
        chrg_vocab_size: size of the vocabulary of charges    == number of different charges    == |C| + 1
        max_number_of_atoms: maximum number of atoms in any molecule
        sequence_length: size of the neighbourhood, i.e. k
        learning_rate: learning rate
        """

        self.d_atom = d_atom
        self.d_dist = d_dist
        self.d_chrg = d_chrg
        self.cf = cf
        self.h = h
        self.atom_vocab_size = atom_vocab_size
        self.dist_vocab_size = dist_vocab_size
        self.chrg_vocab_size = chrg_vocab_size
        self.max_number_of_atoms = max_number_of_atoms
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        self._init_model()

    def train_batch(self, batch_atom, batch_dist, batch_chrg, batch_ys):
        """
        Train one batch.
        """
        self.sess.run(self.train_step,
                      feed_dict={self.x_atom: batch_atom, self.x_dist: batch_dist, self.x_chrg: batch_chrg,
                                 self.y_: batch_ys})

    def get_pred(self, batch_atom, batch_dist, batch_chrg):
        """
        Predict one batch.
        """
        pred = self.sess.run(self.y,
                             feed_dict={self.x_atom: batch_atom, self.x_dist: batch_dist, self.x_chrg: batch_chrg})
        return pred

    def _init_model(self):
        # input variable atom types, environment descriptor for each atom
        self.x_atom = tf.placeholder(tf.int32, shape=[None, self.max_number_of_atoms, self.sequence_length])
        # inut variable distances, environment descriptor for each atom
        self.x_dist = tf.placeholder(tf.int32, shape=[None, self.max_number_of_atoms, self.sequence_length])
        # input variable charges, environment descriptor for each atom
        self.x_chrg = tf.placeholder(tf.int32, shape=[None, self.max_number_of_atoms, self.sequence_length])
        # input variable true classes
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])

        # embedding of atom types, corresponds to W^{atm}
        # this is an R^{|A| x d_atom} matrix
        self.W_atom = tf.Variable(tf.random_uniform([self.atom_vocab_size - 1, self.d_atom], -0.1, 0.1))
        # padding to a R^{(|A| + 1) x d_atom} with 0 to account for
        # 'invalid' atom entries (environments for non-present atoms are filled with |A|+1)
        self.W_atom_padded = tf.pad(self.W_atom, [[0, 1], [0, 0]], 'CONSTANT')
        # lookup the embedding => (?, max_nof_atoms, k, d_atom) tensor
        self.E_atom = tf.nn.embedding_lookup(self.W_atom_padded, self.x_atom)
        print(self.E_atom)
        # reshape such that all entries of the environment are concatenated
        # => (?, max_nof_atoms, k * d_atom) tensor
        # last dimension (?, i, :) corresponds to vector z_{atm} from the paper
        self.E_atom_rs = tf.reshape(self.E_atom, [-1, self.max_number_of_atoms, self.sequence_length * self.d_atom])
        print(self.E_atom_rs)

        # embedding of atom types, corresponds to W^{dist}
        # this is an R^{|D| x d_dist} matrix
        self.W_dist = tf.Variable(tf.random_uniform([self.dist_vocab_size - 1, self.d_dist], -0.1, 0.1))
        # padding to a R^{(|D| + 1) x d_dist} with 0 to account for
        # 'invalid' distance entries (environments for non-present atoms are filled with |D|+1)
        self.W_dist_padded = tf.pad(self.W_dist, [[0, 1], [0, 0]], 'CONSTANT')
        # lookup the embedding => (?, max_nof_atoms, k, d_dist) tensor
        self.E_dist = tf.nn.embedding_lookup(self.W_dist_padded, self.x_dist)
        print(self.E_dist)
        # reshape such that all entries of the environment are concatenated
        # => (?, max_nof_atoms, k * d_dist) tensor
        # last dimension (?, i, :) corresponds to vector z_{dist} from the paper
        self.E_dist_rs = tf.reshape(self.E_dist, [-1, self.max_number_of_atoms, self.sequence_length * self.d_dist])
        print(self.E_dist_rs)

        # embedding of atom types, corresponds to W^{chrg}
        # this is an R^{|C| x d_chrg} matrix
        self.W_chrg = tf.Variable(tf.random_uniform([self.chrg_vocab_size - 1, self.d_chrg], -0.1, 0.1))
        # padding to a R^{(|C| + 1) x d_chrg} with 0 to account for
        # 'invalid' distance entries (environments for non-present atoms are filled with |C|+1)
        self.W_chrg_padded = tf.pad(self.W_chrg, [[0, 1], [0, 0]], 'CONSTANT')
        # lookup the embedding => (?, max_nof_atoms, k, d_chrg) tensor
        self.E_chrg = tf.nn.embedding_lookup(self.W_chrg_padded, self.x_chrg)
        print(self.E_chrg)
        # reshape such that all entries of the environment are concatenated
        # => (?, max_nof_atoms, k * d_chrg) tensor
        # last dimension (?, i, :) corresponds to vector z_{chrg} from the paper
        self.E_chrg_rs = tf.reshape(self.E_chrg, [-1, self.max_number_of_atoms, self.sequence_length * self.d_chrg])
        print(self.E_chrg_rs)

        # TODO; here the amino acid descriptor could be added

        z_size = (self.d_atom + self.d_dist + self.d_chrg) * self.sequence_length
        # concatenate along the last dimension
        # => (?, max_nof_atoms, k * (d_atom + d_dist + d_chrg)) tensor
        # last dimension (?, i, :) corresponds to vector z_i from the paper
        # (of course currently without the amino acid part)
        self.E_concat = tf.concat([self.E_atom_rs, self.E_dist_rs, self.E_chrg_rs], 2)
        """
        z_size = (self.d_atom) * self.sequence_length
        self.E_concat = tf.concat(2, [self.E_atom_rs])
        """

        # added the missing input channel dimension for convolution (1)
        self.E_concat_exp = tf.expand_dims(self.E_concat, -1)
        print(self.E_concat_exp)

        # define filter shape for convolution
        # [filter_height = 1, filter_width = k * (d_atom + d_dist + d_chrg), input_channels = 1, output_channels = cf]
        filter_shape = [1, z_size, 1, self.cf]
        # print(filter_shape)
        self.W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        print(self.W1)
        self.b1 = tf.Variable(tf.constant(0.1, shape=[self.cf]))
        # perform convolution (stride = 1 in each direction, we don't actually move the filter along the width)
        # results in a (?, max_nof_atoms, 1, cf) tensor (it's convoluted now... ;)
        self.Conv = tf.nn.conv2d(self.E_concat_exp, self.W1, strides=[1, 1, 1, 1], padding='VALID')
        print(self.Conv)
        # add bias and use hyperbolic tanges activation function as specified in the paper
        # now (?, i, 1, :) contains U[:,i] from the paper
        self.U = tf.tanh(tf.nn.bias_add(self.Conv, self.b1))
        print(self.U)
        # perform max-polling along dimension 1 (the dimension of the atoms)
        # resulting in a (?, 1, 1, cf) tensor
        self.r = tf.nn.max_pool(self.U, ksize=[1, self.max_number_of_atoms, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID')
        print(self.r)

        # we now reshape to remove the now-meaningsless dimensions
        # resulting in a (?, cf) tensor, which corresponds to r from the paper
        self.r_flat = tf.reshape(self.r, [-1, self.cf])
        print(self.r_flat)

        # the rest is straight forward
        # first, the hidden layer, W_2 \in R^{h x cf}
        self.W2 = tf.Variable(tf.truncated_normal([self.cf, self.h], stddev=0.1))
        # with bias
        self.b2 = tf.Variable(tf.constant(0.1, shape=[self.h]))

        print(self.W2)
        self.S = tf.matmul(self.r_flat, self.W2) + self.b2
        print(self.S)

        # and then the output layer, W_r \in R^{2 x h}
        self.W3 = tf.Variable(tf.truncated_normal([self.h, 2], stddev=0.1))
        self.b3 = tf.Variable(tf.constant(0.1, shape=[2]))
        print(self.W3)

        self.y_raw = tf.matmul(self.S, self.W3) + self.b3

        # train using cross entropy (or log-loss) as in the paper
        # CAUTION: must be used on the raw scores, peforms softmax internally
        self.cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.y_, 1), logits=self.y_raw))

        # softmax function for actual output
        self.y = tf.nn.softmax(self.y_raw)

        # one training step with Gradient Descent
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)

        init = tf.global_variables_initializer()

        self.sess = tf.Session(
            config=tf.ConfigProto(intra_op_parallelism_threads=1)
        )
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())


def main():

    parser = argparse.ArgumentParser('Trains and evaluates the DeepVS convolutional neural network.')
    parser.add_argument('--input', type=str, required=True,
                        help='input file')
    parser.add_argument('--test_id', type=int, required=True,
                        help='0-based ID of subset used for testing. Each ID corresponds to target protein')
    parser.add_argument('--exclude', type=str, default=None,
                        help='json file with exclusion information')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')

    args = parser.parse_args()

    test_id = args.test_id

    print("test_id ", test_id)

    df = pd.read_csv(args.input, header=0)

    # extract the columns from input data
    atom_columns = [c for c in df.columns if c.startswith('type_')]
    dist_columns = [c for c in df.columns if c.startswith('dist_')]
    chrg_columns = [c for c in df.columns if c.startswith('chrg_')]
    sequence_length = max([int(c.split('_')[2]) for c in atom_columns]) + 1  # k_c
    max_number_of_atoms = max([int(c.split('_')[1]) for c in atom_columns]) + 1
    print(sequence_length, max_number_of_atoms)

    # values as in the paper
    d_atom = 200
    d_dist = 200
    d_chrg = 200
    cf = 400
    h = 50
    learning_rate = 0.075
    batch_size = 20
    epochs = args.epochs
    assert(epochs > 0)

    # vocabulary sizes are the maximum values observed
    # (the last one is always invalid, therefor always present)
    # + 1 (because of 0-based)
    atom_vocab_size = int(df.iloc[0][atom_columns[-1]]) + 1
    dist_vocab_size = int(df.iloc[0][dist_columns[-1]]) + 1
    chrg_vocab_size = int(df.iloc[0][chrg_columns[-1]]) + 1

    print('detected atom_vocab_size', atom_vocab_size)
    print('detected dist_vocab_size', dist_vocab_size)
    print('detected chrg_vocab_size', chrg_vocab_size)

    target_names = sorted(list(set(df['code'])))
    print(target_names)

    test_target = target_names[test_id]
    print('TEST-TARGET: {}'.format(test_target))

    training_targets = [c for c in target_names if c != test_target]

    if args.exclude:
        with open(args.exclude, 'r') as infile:
            excludes = json.load(infile)

        for c in target_names:
            print(c)
            assert c in excludes

        assert len(target_names) == len(excludes)

        for e in excludes.values():
            for c in e:
                assert c in target_names

        training_excludes = excludes[test_target]
    else:
        training_excludes = []

    print('excluding', training_excludes)

    training_targets = [c for c in training_targets if c not in training_excludes]

    print('training set', training_targets)

    training_df = df[df.code.isin(training_targets)]
    test_df = df[df.code == test_target]

    # prepare the test data
    test_atom_values = test_df[atom_columns].astype(int).values
    test_atom_values = test_atom_values.reshape([len(test_atom_values), max_number_of_atoms, sequence_length])
    test_dist_values = test_df[dist_columns].astype(int).values
    test_dist_values = test_dist_values.reshape([len(test_dist_values), max_number_of_atoms, sequence_length])
    test_chrg_values = test_df[chrg_columns].astype(int).values
    test_chrg_values = test_chrg_values.reshape([len(test_chrg_values), max_number_of_atoms, sequence_length])
    test_y_values = test_df[['inactive', 'active']].astype(int).values

    # generate model
    model = DeepVSModel(d_atom, d_dist, d_chrg, cf, h, atom_vocab_size, dist_vocab_size, chrg_vocab_size,
                        max_number_of_atoms, sequence_length, learning_rate)

    for e in range(epochs):
        print('starting epoch {}'.format(e))

        # shuffle training values
        training_df = training_df.sample(frac=1)

        # extract descriptors into np arrays
        training_atom_values = training_df[atom_columns].astype(int).values
        training_atom_values = training_atom_values.reshape(
            [len(training_atom_values), max_number_of_atoms, sequence_length])
        training_dist_values = training_df[dist_columns].astype(int).values
        training_dist_values = training_dist_values.reshape(
            [len(training_dist_values), max_number_of_atoms, sequence_length])
        training_chrg_values = training_df[chrg_columns].astype(int).values
        training_chrg_values = training_chrg_values.reshape(
            [len(training_chrg_values), max_number_of_atoms, sequence_length])
        training_y_values = training_df[['inactive', 'active']].astype(int).values

        # calculate the ratio between positive and negative samples
        num_neg_examples = sum(training_y_values[:, 0])
        num_pos_examples = sum(training_y_values[:, 1])
        assert num_neg_examples + num_pos_examples == len(training_y_values)
        ratio = float(num_neg_examples) / float(num_pos_examples)
        print('detected ratio neg/pos of', ratio)

        # train batches
        for i in range(int(math.ceil(len(training_y_values) / batch_size))):
            batch_atom = training_atom_values[batch_size * i: batch_size * (i + 1)]
            batch_dist = training_dist_values[batch_size * i: batch_size * (i + 1)]
            batch_chrg = training_chrg_values[batch_size * i: batch_size * (i + 1)]
            batch_ys = training_y_values[batch_size * i: batch_size * (i + 1)]
            model.train_batch(batch_atom, batch_dist, batch_chrg, batch_ys)
            if i % 25 == 0:
                print(i)
                y_pred = model.get_pred(batch_atom, batch_dist, batch_chrg)
                y_pred_bin = np.argmax(y_pred, axis=1)
                print('batch training CA', sklearn.metrics.accuracy_score(y_true=batch_ys[:, 1], y_pred=y_pred_bin))
                try:
                    auc_sklearn = sklearn.metrics.roc_auc_score(y_true=batch_ys, y_score=y_pred)
                    auc_sklearn1 = sklearn.metrics.roc_auc_score(y_true=batch_ys[:, 0], y_score=y_pred[:, 0])
                    auc_sklearn2 = sklearn.metrics.roc_auc_score(y_true=batch_ys[:, 1], y_score=y_pred[:, 1])
                except ValueError as e:
                    print(e)
                    auc_sklearn = None
                    auc_sklearn1 = None
                    auc_sklearn2 = None
                print('batch training AUC', auc_sklearn, (auc_sklearn1, auc_sklearn2,))

        y_pred = np.empty(shape=(0, 2))

        # predict batches of test data (to save memory)
        for i in range(int(math.ceil(len(test_y_values) / batch_size))):
            batch_atom = test_atom_values[batch_size * i: batch_size * (i + 1)]
            batch_dist = test_dist_values[batch_size * i: batch_size * (i + 1)]
            batch_chrg = test_chrg_values[batch_size * i: batch_size * (i + 1)]

            y_pred_local = model.get_pred(batch_atom, batch_dist, batch_chrg)

            # collect the prediction results)
            y_pred = np.concatenate((y_pred, y_pred_local,), axis=0)

        # Analyzing predictions and calc metrics
        y_pred_bin = np.argmax(y_pred, axis=1)
        test_y_values_bin = np.argmax(test_y_values, axis=1)
        sum_pos_pred = sum(y_pred_bin)
        print('final test positive predicted', sum_pos_pred)
        sum_neg_pred = len(y_pred_bin) - sum_pos_pred
        print('final test negative predicted', sum_neg_pred)
        tp = sum([1 for p, a in zip(y_pred_bin, test_y_values_bin) if p == a and p == 1])
        sens = tp / float(sum(test_y_values[:, 1]))
        print('final test sens', sens)
        tn = sum([1 for p, a in zip(y_pred_bin, test_y_values_bin) if p == a and p == 0])
        spec = tn / float(sum(test_y_values[:, 0]))
        print('final test spec', spec)

        m_auc = my_auc(test_y_values[:, 1], y_pred[:, 1])
        print('final test AUC my', m_auc)

        ef_2 = my_ef(test_y_values[:, 1], y_pred[:, 1], 0.02)
        ef_20 = my_ef(test_y_values[:, 1], y_pred[:, 1], 0.2)

        print('final test EF_2/EF_20', (ef_2, ef_20,))

        try:
            auc_sklearn1 = sklearn.metrics.roc_auc_score(y_true=test_y_values[:, 0], y_score=y_pred[:, 0])
            auc_sklearn2 = sklearn.metrics.roc_auc_score(y_true=test_y_values[:, 1], y_score=y_pred[:, 1])
        except ValueError:
            auc_sklearn1 = None
            auc_sklearn2 = None

        print('final test ACC', sklearn.metrics.accuracy_score(y_true=test_y_values[:, 1], y_pred=y_pred_bin))

        auc_sklearn = sklearn.metrics.roc_auc_score(y_true=test_y_values, y_score=y_pred)
        print('final test AUC sklearn', auc_sklearn, (auc_sklearn1, auc_sklearn2,))



if __name__ == "__main__":
    main()
