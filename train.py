__author__ = 'dismael'

"""
Training the model 

model options:

    model parameters:
    N_USERS: (int) number of users in the model
    N_ITEM: (int) number of items in the model
    D_EMBED: (int) dimension of embedding vectors
    N_NEG_SAMPLES: (int) number of negative items to sample per minibatch
    LOSS: (string) loss function to use.
        'explicit_mf': explicit matrix factorization, i.e., mean squared error
        'softmax': sampled softmax cross-entropy
    WEIGHTED: (bool) weight loss by score of each example
    C: (float) regularization weight to apply L2 regularization to item embeddings 
            (don't use--kills sparsity. use weight decay instead)
    USER_EMBEDDINGS_WD_MULT: (float) multiplier for L2 regularization weight for customer embeddings
    ITEM_EMBEDDINGS_WD_MULT: (float) multiplier for L2 regularization weight for item embeddings
    USER_EMBEDDINGS_INIT: (string) path to a customer embeddings matrix (.npy) to use as initialization
    ITEM_EMBEDDINGS_INIT: (string) path to a item embeddings matrix (.npy) to use as initialization
    TRAIN_DIR: (string) path to directory with training data
    N_DATA_COL: (int) number of columns in training data
    FEAT_INDS: (list of ints) column indices to use as features
    FEAT_NAMES: (list of strings) names of features

    training parameters:
    RESTORE: (bool) restore the model from a previously saved checkpoint?
    SPARSE: (bool) use sparse embeddings? 
    BATCH_SIZE: (int) number of examples per minibatch
    N_GPU: (int) number of GPUs to use
    OPTIMIZER: (string) type of optimizer to use
        'sgd': stochastic gradient descent
    OPT_PARAMS: (dict) dicitonary of parameters for the optimizer 
        (depends on the type of optimizer what is required)
        e.g., for sgd: {'learning_rate': 10, 'wd':1e-8}
    N_EPOCH: (int) number of epochs to train for
    PROFILE: (bool) run the profiler info for an iteration? 

"""
import mxnet as mx
import numpy as np
import os, glob, shutil, logging, json, time, re
from read_data import DataIterator
from make_output_layer import pairwise_inner_prod
from make_embeddings_model import embedding_model


def load_opts(log_dir):
    """ Load json with options dict """
    with open('{}/opts.json'.format(log_dir)) as fi:
        opts = json.load(fi)
    return opts


def print_opts(opts):
    """ Print options dictionary """
    for k in opts.keys():
        print('{}: {}'.format(k, opts[k]))


def prepare_log_dir(opts):
    """ Prepare log directory for training """
    if not opts['RESTORE']:
        # if not restoring the model, clear the log directory
        if os.path.exists(opts['LOG_DIR']):
            shutil.rmtree(opts['LOG_DIR'])
        os.makedirs(opts['LOG_DIR'])
        # save opts
        with open(os.path.join(opts['LOG_DIR'], 'opts.json'), 'w') as of:
            json.dump(opts, of)
    else:
        # if restoring, read the options from previous training
        with open(os.path.join(opts['LOG_DIR'], 'opts.json')) as iif:
            opts = json.load(iif)
            opts['RESTORE'] = True
    return opts


class CustomSpeedometer(object):
    """ Logs training speed and evaluation metrics periodically.
    Inputs:
        batch_size (int): Batch size of data
        log_file: path to write log data
        frequent (int): Specifies how frequently training speed and evaluation metrics must be logged.
            Default behavior is to log once every 100 batches.
        auto_reset (bool): Reset the evaluation metrics after each log.
    Example output
    -------
    Epoch[0] Batch [144100]       Speed: 162243.58 samples/sec    loss=9.526037
    Epoch[0] Batch [144200]       Speed: 160198.88 samples/sec    loss=9.654321
    Epoch[0] Batch [144300]       Speed: 166761.27 samples/sec    loss=9.605808
    Epoch[0] Batch [144400]       Speed: 159008.61 samples/sec    loss=9.69425
    """

    def __init__(self, batch_size, log_file, frequent=100, auto_reset=True, prof_file=None,
                 profile=False):
        self.batch_size = batch_size
        self.log_file = log_file
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset
        self.prof_file = prof_file
        self.profile = profile

    def __call__(self, param):
        """Callback to show speed"""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.profile:
            # log profiling info for step 13 (an arbitrary step)
            if count == 13:
                mx.profiler.profiler_set_config(mode='all', filename=self.prof_file)
                print('writing profiling info to {}'.format(self.prof_file))
                mx.profiler.profiler_set_state('run')
            elif count == 14:
                mx.profiler.profiler_set_state('stop')
                mx.profiler.dump_profile()
                print('stopping profiler')

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec'
                    msg += '\t%s=%f' * len(name_value)
                    logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                    # log eval metric
                    with open(self.log_file, 'a') as fo:
                        fo.write('{},{},{}\n'.format(param.epoch, count, name_value[0][1]))
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def custom_checkpoint(log_dir, period=1):
    """ A callback that saves a model checkpoint every few epochs.
    Each checkpoint is made up of a couple of binary files: a model description file and a
    parameters (weights and biases) file. The model description file is named
    `prefix`--symbol.json and the parameters file is named `prefix`-`epoch_number`.params
    Inputs:
        log_dir: (str) Directory to save checkpoints
        period : (int) Interval (number of epochs) between checkpoints. Default `period` is 1.
    Outputs:
        A callback function that can be passed as `epoch_end_callback` to fit.
    Start training with [cpu(0)]
    Epoch[0] Resetting Data Iterator
    Epoch[0] Time cost=0.100
    Saved checkpoint to "mymodel-0001.params"
    Epoch[1] Resetting Data Iterator
    Epoch[1] Time cost=0.060
    Saved checkpoint to "mymodel-0002.params"
    """
    period = int(max(1, period))  # smallest period is 1 epoch

    def _callback(iter_no, sym, arg, aux):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            # first remove old checkpoints
            for f in glob.glob(os.path.join(log_dir, 'ckpt*params')):
                os.remove(f)
            # save the latest checkpoint
            mx.model.save_checkpoint(os.path.join(log_dir, 'ckpt'), iter_no + 1, sym, arg, aux)

    return _callback


def create_initializer(opts):
    """ Create initializer for embeddings """
    params = {}
    # load customer embeddings and put in params dictionary
    if opts.get('USER_EMBEDDINGS_INIT', False):
        ce = np.load(opts['USER_EMBEDDINGS_INIT'])
        if ce.shape[1] != opts['D_EMBED']:
            raise Exception(('User embedding initializer ({}) does not match embedding dimension'
                             '({})').format(ce.shape[1],
                                            opts['D_EMBED']))
        if ce.shape[0] < opts['N_USERS']:

            print('Adding {} users to the model'.format(opts['N_USERS'] - ce.shape[0]))
            ce_rand = np.random.normal(scale=0.1, size=(opts['N_USERS'], opts['D_EMBED']))
            ce_rand[:ce.shape[0], :] = ce
            ce = ce_rand

        elif ce.shape[0] > opts['N_USERS']:
            raise Exception(('User embedding initializer ({}) has too many users'
                             '({})').format(ce.shape[0],
                                            opts['N_USERS']))

        params['user_embeddings'] = ce

    # load item embeddings and put in params dictionary
    if opts.get('ITEM_EMBEDDINGS_INIT', False):
        te = np.load(opts['ITEM_EMBEDDINGS_INIT'])
        if te.shape[1] != opts['D_EMBED']:
            raise Exception(('Item embedding initializer ({}) does not match embedding dimension'
                             '({})').format(te.shape[1],
                                            opts['D_EMBED']))
        if te.shape[0] < opts['N_ITEM']:

            print('Adding {} items to the model'.format(opts['N_ITEM'] - te.shape[0]))
            te_rand = np.random.normal(scale=0.1, size=(opts['N_ITE,'], opts['D_EMBED']))
            te_rand[:te.shape[0], :] = te
            te = te_rand

        elif te.shape[0] > opts['N_ITEM']:
            raise Exception(('Item embedding initializer ({}) has too many items'
                             '({})').format(te.shape[0], opts['N_ITEM']))

        params['item_embeddings'] = te
    if params:
        initer = mx.init.Load(params, default_init=mx.init.Normal(0.1), verbose=True)
    else:
        initer = mx.init.Normal(0.1)
    return initer


def get_context(n_gpu):
    if n_gpu > 0:
        ctx = [mx.gpu(k) for k in range(n_gpu)]
    else:
        ctx = mx.cpu(0)
    print('training on: {}'.format(ctx))
    return ctx

def train(opts, frequent=10):
    mx.random.seed(0)
    """ Train a model """
    train_data = DataIterator(opts['TRAIN_DIR'],
                              batch_size=opts['BATCH_SIZE'],
                              n_col=opts['N_DATA_COL'],
                              feat_names=opts.get('FEAT_NAMES', []),
                              feat_inds=opts.get('FEAT_INDS', []),
                              use_label=opts['WEIGHTED']
                              )
    # train_data = mx.io.PrefetchingIter(train_data) # this doesn't seem to help speed
    data_names, _ = zip(*train_data._provide_data)
    label_names, _ = zip(*train_data._provide_label)

    loss = embedding_model(opts)

    # intializer
    initer = create_initializer(opts)

    # GPUs
    ctx = get_context(opts['N_GPU'])

    # model
    model = mx.mod.Module(symbol=loss,
                          context=ctx,
                          data_names=data_names,
                          label_names=label_names)

    logging.basicConfig(level=logging.DEBUG)

    model.fit(train_data,
              optimizer=opts['OPTIMIZER'],
              eval_metric='Loss',
              optimizer_params=opts['OPT_PARAMS'],
              num_epoch=opts['N_EPOCH'],
              initializer=initer,
              batch_end_callback=CustomSpeedometer(batch_size=opts['BATCH_SIZE'],
                                                   log_file=os.path.join(opts['LOG_DIR'], 'train'),
                                                   frequent=frequent,
                                                   prof_file=os.path.join(opts['LOG_DIR'], 'profile_output.json'),
                                                   profile=opts.get('PROFILE', False)),
              epoch_end_callback=custom_checkpoint(log_dir=opts['LOG_DIR']))


def get_embeddings(log_dir, opts, epoch=None):
    """ load embeddings from model checkpoint """
    if epoch is None:
        epoch = get_epoch(opts['LOG_DIR'])
    print('loading embeddings from epoch {} in {}'.format(epoch, log_dir))
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(log_dir, 'ckpt'), epoch=epoch)
    user_embeddings = arg_params['user_embeddings'].asnumpy()
    item_embeddings = arg_params['item_embeddings'].asnumpy()
    return user_embeddings, item_embeddings


def get_scores(opts, test_data, log_dir, epoch=None, test_batch_size=10, gpu=False):
    """ Build network for prediction model """
    test_data_iter = mx.io.NDArrayIter(np.array(test_data),
                                       batch_size=test_batch_size,
                                       data_name='user')

    # load trained embeddings
    if epoch is None:
        epoch = get_epoch(opts['LOG_DIR'])
    print('loading checkpoint from epoch {} in {}'.format(epoch, log_dir))
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(opts['LOG_DIR'], 'ckpt'), epoch=epoch)
    y = prediction_network(opts, sym, arg_params, aux_params)

    if gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)
    pred_model = mx.mod.Module(symbol=y, context=ctx, data_names=['user'], label_names=None)
    pred_model.bind(for_training=False, data_shapes=[('user', (test_batch_size,))])
    pred_model.set_params(arg_params, aux_params, allow_missing=True)
    print('making predictions for {} users'.format(len(test_data)))
    scores = pred_model.predict(test_data_iter)
    return scores


def prediction_network(opts, sym, arg_params, aux_params):
    """ Build network for prediction model """
    # create forward network
    all_layers = sym.get_internals()
    user_embeddings = all_layers['user_embeddings']
    item_embeddings = all_layers['item_embeddings']

    # inputs
    user = mx.sym.var('user')
    item = mx.sym.arange(0, opts['N_ITEMS'], name='item', dtype='int32')

    if type(arg_params['user_embeddings']) == mx.ndarray.sparse.RowSparseNDArray:
        user_embedded = mx.sym.contrib.SparseEmbedding(data=user,
                                                       weight=user_embeddings,
                                                       input_dim=opts['N_USERS'],
                                                       output_dim=opts['D_EMBED'])
    else:
        user_embedded = mx.sym.Embedding(data=user,
                                         weight=user_embeddings,
                                         input_dim=opts['N_USERS'],
                                         output_dim=opts['D_EMBED'])

    if type(arg_params['item_embeddings']) == mx.ndarray.sparse.RowSparseNDArray:
        item_embedded = mx.sym.contrib.SparseEmbedding(data=item,
                                                       weight=item_embeddings,
                                                       input_dim=opts['N_ITEMS'],
                                                       output_dim=opts['D_EMBED'])
    else:
        item_embedded = mx.sym.Embedding(data=item,
                                         weight=item_embeddings,
                                         input_dim=opts['N_ITEMS'],
                                         output_dim=opts['D_EMBED'])

    y = pairwise_inner_prod(user_embedded, item_embedded)
    return y


def get_epoch(log_dir):
    # get maximum epoch in log directory
    matches = [re.search('ckpt-(\d+).params', d) for d in os.listdir(log_dir)]
    ep = [int(m.group(1)) if m else -1 for m in matches]
    return max(ep)
