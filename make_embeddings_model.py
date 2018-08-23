__author__ = 'dismael'

import mxnet as mx
import numpy as np
from make_output_layer import output_layer, l2_regularization


def embedding_model(opts):
    """ Build network for embedding model """

    # inputs
    user = mx.sym.var('user')
    item = mx.sym.var('item')
    score = mx.sym.var('score')

    # embeddings
    if opts.get('SPARSE', True):
        # user embeddings
        user_embeddings = mx.sym.var('user_embeddings',
                                     shape=(opts['N_USERS'], opts['D_EMBED']),
                                     init=mx.init.Normal(0.1),
                                     stype='row_sparse',
                                     lr_mult=1,
                                     wd_mult=opts.get('USER_EMBEDDINGS_WD_MULT', 0)
                                     )
        user_embedded = mx.sym.contrib.SparseEmbedding(data=user,
                                                       weight=user_embeddings,
                                                       input_dim=opts['N_USERS'],
                                                       output_dim=opts['D_EMBED'],
                                                       name='user_embedded')
        # item embeddings
        item_embeddings = mx.sym.var('item_embeddings',
                                     shape=(opts['N_ITEMS'], opts['D_EMBED']),
                                     init=mx.init.Normal(0.1),
                                     stype='row_sparse',
                                     lr_mult=1,
                                     wd_mult=opts.get('item_EMBEDDINGS_WD_MULT', 0)
                                     )
        item_embedded = mx.sym.contrib.SparseEmbedding(data=item,
                                                       weight=item_embeddings,
                                                       input_dim=opts['N_ITEMS'],
                                                       output_dim=opts['D_EMBED'],
                                                       name='item_embedded')

    else:
        # user embeddings
        user_embeddings = mx.sym.var('user_embeddings',
                                     shape=(opts['N_USERS'], opts['D_EMBED']),
                                     init=mx.init.Normal(0.1),
                                     lr_mult=1,
                                     wd_mult=opts.get('user_EMBEDDINGS_WD_MULT', 0)
                                     )
        user_embedded = mx.sym.Embedding(data=user,
                                         weight=user_embeddings,
                                         input_dim=opts['N_USERS'],
                                         output_dim=opts['D_EMBED'])
        # item embeddings
        item_embeddings = mx.sym.var('item_embeddings',
                                     shape=(opts['N_ITEMS'], opts['D_EMBED']),
                                     init=mx.init.Normal(0.1),
                                     lr_mult=1,
                                     wd_mult=opts.get('item_EMBEDDINGS_WD_MULT', 0)
                                     )
        item_embedded = mx.sym.Embedding(data=item,
                                         weight=item_embeddings,
                                         input_dim=opts['N_ITEMS'],
                                         output_dim=opts['D_EMBED'])

    arg_shapes, out_shapes, aux_shapes = user_embeddings.infer_shape()
    print('user_embeddings dim={}'.format(out_shapes[0]))
    arg_shapes, out_shapes, aux_shapes = item_embeddings.infer_shape()
    print('item_embeddings dim={}'.format(out_shapes[0]))
    arg_shapes, out_shapes, aux_shapes = user_embedded.infer_shape(user=(opts['BATCH_SIZE'],))
    print('user_embedded dim={}'.format(out_shapes[0]))
    arg_shapes, out_shapes, aux_shapes = item_embedded.infer_shape(item=(opts['BATCH_SIZE'],))
    print('item_embedded dim={}'.format(out_shapes[0]))

    if opts['N_NEG_SAMPLES'] > 0:
        # putting random sampling in the model
        neg_samples = mx.sym.random_uniform(low=0, high=opts['N_item'] - 1,
                                            shape=(opts['N_NEG_SAMPLES'],))
        # negative samples lookup
        if opts.get('SPARSE', True):
            neg_embedded = mx.sym.contrib.SparseEmbedding(data=neg_samples,
                                                          weight=item_embeddings,
                                                          input_dim=opts['N_item'],
                                                          output_dim=opts['D_EMBED'],
                                                          name='neg_embedded')
        else:
            neg_embedded = mx.sym.Embedding(data=neg_samples,
                                            weight=item_embeddings,
                                            input_dim=opts['N_item'],
                                            output_dim=opts['D_EMBED'])

        arg_shapes, out_shapes, aux_shapes = neg_embedded.infer_shape()
        print('neg_embedded dim={}'.format(out_shapes[0]))
    else:
        neg_embedded = None

    # get the loss
    loss = output_layer(user_embedded, item_embedded, neg_embedded, score, opts)

    # add regularization
    if opts.get('C', None) is not None:
        reg_loss = l2_regularization(item_embeddings, opts['C'])
        loss = mx.sym.Group([loss, reg_loss])
        return loss
    else:
        return loss
