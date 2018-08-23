__author__ = 'dismael'
import mxnet as mx





class WeightedSoftmaxCrossEntropyLoss(mx.operator.useromOp):
    """ An output layer that calculates gradient for cross-entropy loss
    - sum_i (y_i * log(p_i)). or - log(p[y]) where y is the label
    for label "y" and prediction "p".
    Currently only implemented for data where the true class is at index 0.
    """
    # ref: http://mxnet.io/how_to/new_op.html
    eps = 1e-6  # Avoid -inf when taking log(0)

    def forward(self, is_train, req, in_data, out_data, aux):
        # Shapes:
        #  b = minibatch size
        #  d = number of dimensions

        data = in_data[0]  # shape=(b,d)
        score = in_data[1]  # shape=(b,)
        b = data.shape[0]

        log_probs = mx.nd.log_softmax(data, axis=1)
        # the true class is always in position 0
        cross_entropy = - mx.nd.slice_axis(log_probs, axis=1, begin=0, end=1).reshape((b,))
        # weight cross entropy by the score
        x = score * cross_entropy
        self.assign(out_data[0], req[0], x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # grad  = p_i - y_i where p_i is softmax output and y_i is (binary) label
        data = in_data[0]  # shape=(b,d)
        score = in_data[1]  # shape=(b,)
        b = data.shape[0]
        d = data.shape[1]

        probs = mx.nd.softmax(data, axis=1)
        y = mx.nd.one_hot(mx.nd.zeros((b,)), d)
        score = mx.nd.expand_dims(score, axis=1)
        score = mx.nd.tile(score, reps=(1, d))
        grad = score * (probs - y)
        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("WeightedSoftmaxCrossEntropyLoss")
class WeightedSoftmaxCrossEntropyProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(WeightedSoftmaxCrossEntropyProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['preds']

    def create_operator(self, ctx, shapes, dtypes):
        return WeightedSoftmaxCrossEntropyLoss()

    def infer_shape(self, in_shape):
        if in_shape[0][0] != in_shape[1][0]:
            raise ValueError("First dimension of inputs differ. data:%s. label:%s. must be same"
                             % (str(in_shape[0]), str(in_shape[1])))
        output_shape = in_shape[1]
        return in_shape, [output_shape], []



def l2_regularization(var, reg_weight):
    """ Create a loss tensor for L2 regularization of a variable
    Inputs:
        var: variable to regularize
        reg_weight: weight to apply to regularization loss
    Output:
        symbol with loss
    """
    l2_loss = mx.sym.sum(mx.sym.square(var))
    reg_loss = mx.sym.MakeLoss(l2_loss * (reg_weight / 2))
    return reg_loss

def matched_inner_prod(mat1, mat2):
    """ Inner product between matched rows of 2 matrices,
    i.e., i^th row of output matrix is the inner product of i^th row of mat1 and i^th row of mat2 """
    out = mat1 * mat2  # elementwise product
    out = mx.sym.sum_axis(data=out, axis=1)  # sum over columns
    out = mx.sym.flatten(data=out)
    return out


def pairwise_inner_prod(mat1, mat2):
    """ Inner product between 2 matrices, i.e., element i,j of output matrix is the
    inner product of i^th row of mat1 and j^th row of mat2 """
    return mx.sym.dot(mat1, mat2, transpose_b=True)

def mf_layer(user_embedded, item_embedded, score, opts):
    """ Simple matrix factorization layer"""
    # predict by the inner product, which is elementwise product and then sum
    pred = matched_inner_prod(user_embedded, item_embedded)
    # loss layer
    loss = mx.sym.LinearRegressionOutput(data=pred, label=score)
    return loss


def weighted_softmax_layer(user_embedded, item_embedded, neg_embedded, score, opts):
    # predict by the inner product, which is elementwise product and then sum
    pos_pred = matched_inner_prod(user_embedded, item_embedded)
    neg_pred = pairwise_inner_prod(user_embedded, neg_embedded)
    pred = mx.sym.concat(pos_pred, neg_pred, dim=1)
    # loss layer - use custom op defined below
    loss = mx.symbol.Custom(data=pred,
                            label=score,
                            name='loss',
                            op_type='WeightedSoftmaxCrossEntropyLoss')
    return loss

def softmax_layer(user_embedded, item_embedded, neg_embedded, score, opts):
    """ Classification layer with sampled softmax cross entropy
    Inputs:
        user_embedded: embedded user vectors
        item_embedded: embedded item vectors
        neg_embedded: embedded item vectors for negative samples
        score: scores/counts for each example - these should be all zeros
        opts: dictionary with model options
    Output:
        symbol with softmax cross entropy loss
    """
    if neg_embedded is None:
        raise Exception('Softmax output requires number of negative samples > 0')
    # dot prod of matched users and true items
    pos_pred = matched_inner_prod(user_embedded, item_embedded)

    # dot prod of user and all neg items
    neg_pred = pairwise_inner_prod(user_embedded, neg_embedded)

    # concatenate pos and neg horizontally
    data = mx.sym.concat(pos_pred, neg_pred, dim=1)

    # softmax cross entropy
    loss = mx.sym.SoftmaxOutput(data, score)
    arg_shapes, out_shapes, aux_shapes = loss.infer_shape(user=(opts['BATCH_SIZE'],),
                                                          item=(opts['BATCH_SIZE'],),
                                                          score=(opts['BATCH_SIZE'],))
    print('loss dim={}'.format(out_shapes[0]))

    return loss

def output_layer(user_embedded, item_embedded, neg_embedded, score, opts):
    """ Create the output layer
    Inputs:
        user_embedded: embedded user vectors
        item_embedded: embedded item vectors
        neg_embedded: embedded  vectors for negative samples
        score: scores/counts for each example
        opts: dictionary with model options
    Output:
        symbol with loss
    """
    if opts['LOSS'] == 'explicit_mf':
        return mf_layer(user_embedded, item_embedded, score, opts)
    elif opts['LOSS'] == 'softmax':
        if opts['WEIGHTED'] == True:
            # weighted softmax by counts/score
            return weighted_softmax_layer(user_embedded, item_embedded, neg_embedded, score, opts)
        else:
            # unweighted softmax
            return softmax_layer(user_embedded, item_embedded, neg_embedded, score, opts)
    else:
        raise Exception('Unknown loss!')