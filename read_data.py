__author__ = 'dismael'

import mxnet as mx

class DataIterator(mx.io.DataIter):
    """ Iterator to iterate over the training data.
            The first column is user index, the second column is item index,
            the last column in the score the user gives to an item.

        Inputs:
            filepath: path to training data
            batch_size: Number of batches of data to send out
            n_col: number of columns in training data
            feature_names: list of feature names
            feature_inds: list of column indexes to use as additional features
        Output:
            data batch iterator

    for more details please refer to: https://mxnet.incubator.apache.org/tutorials/basic/data.html

    """

    def __init__(self, filepath, batch_size, n_col=3, feature_names=[], feature_inds=[]):
        super().__init__()
        self.batch_size = batch_size
        self.feat_names = feature_names
        self.feat_inds = feature_inds
        self.n_col = n_col
        self.csv_iter = mx.io.CSVIter(data_csv=filepath,
                                      data_shape=(n_col,),
                                      batch_size=batch_size,
                                      round_batch=False)

        if len(feature_names) != len(feature_inds):
            raise Exception('feat_names and feat_inds should be the same length')
        if not set(feature_inds).issubset(set(range(2, n_col - 2))):
            raise Exception('invalid feat_inds')

        data_shapes = [('user', (batch_size,)), ('item', (batch_size,))]
        for feat_name in feature_names:
            data_shapes.append((feat_name, (batch_size,)))

        self._provide_data = data_shapes
        self._provide_label = [('score', (batch_size,))]

    def __iter__(self):
        return self

    def reset(self):
        self.csv_iter.reset()

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        # get next batch of data
        batch = self.csv_iter.next()
        # split columns
        cols = mx.ndarray.split(batch.data[0], num_outputs=self.n_col, axis=1, squeeze_axis=1)
        data = [cols[0], cols[1]]  # first two columns are user and item
        for feat_ind in self.feat_inds:
            data.append(cols[feat_ind])  # add desired features
        if self.use_label:
            # label are the scores - last column
            label = [cols[-1]]
        else:
            # labels are zeros
            label = [mx.ndarray.zeros(shape=(self.batch_size,))]
        return mx.io.DataBatch(data, label)

