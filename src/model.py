from tensorflow import gather_nd
from tensorflow import matmul
from tensorflow.contrib.framework import get_global_step
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.optimizer_v2.adam import AdamOptimizer
from tensorflow.python import expand_dims, shape, SparseTensor, ones, int64, cast, reduce_sum, int32, float32, squeeze, \
    equal, argmax, one_hot, constant
from tensorflow.python.keras.backend import concatenate, to_dense
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.layers.core import dense
from tensorflow.python.ops.math_ops import range
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits
from tensorflow_estimator.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.ops.metrics import mean


def noise_label(labels):
    id = range(shape(labels['click_position'])[0])
    idx = concatenate([expand_dims(cast(id,int64), axis=1), labels['click_position']], axis=1)
    clicked_item = gather_nd(labels['reco'], idx)
    return cast(equal(expand_dims(clicked_item, axis=1), labels['reco']), float32)

def true_label(features, labels):
    return cast(equal(features['anchor'], labels['reco']), float32)

def to_one_hot(scores):
     return one_hot(argmax(scores,axis=1), shape(scores)[1])

def predict_scores(features):
    candidate = dense(features['label'],10)
    anchor = dense(features['anchor_label'], 10)
    scores = matmul(candidate, expand_dims(anchor, axis=1), transpose_b=True)
    return squeeze(scores)

def lookup_positives(scores, click_position):
    num_rows = shape(scores)[0]
    row_idx = expand_dims(range(num_rows), axis=1)
    idx = concatenate([row_idx, cast(click_position, int32)], axis=1)
    return gather_nd(scores, idx)

def create_diffs(positive_scores, scores):
    return expand_dims(positive_scores, axis=1)- scores

def create_label(click_position, num_labels=10):
    num_rows = shape(click_position)[0]
    row_idx = expand_dims(range(num_rows), axis=1)
    idx = concatenate([row_idx, cast(click_position,int32)], axis=1)
    labels = SparseTensor(indices=cast(idx,int64), values=ones([num_rows]),dense_shape=[num_rows, num_labels])
    return ones([num_rows, num_labels]) - to_dense(labels)


def elementwise_loss(labels, logits, mask):
    return sigmoid_cross_entropy_with_logits(labels=labels, logits=logits) * mask

def get_positive_mask():
    pass

def model_fn(features, labels, mode, params):
    scores = predict_scores(features)


    if mode == ModeKeys.INFER:
        return EstimatorSpec(mode, predictions=scores)

    positive_scores = lookup_positives(scores, labels['click_position'])
    logits = create_diffs(positive_scores, scores)
    lbls = create_label(labels['click_position'])
    ele_loss = elementwise_loss(lbls, logits, labels['normal_mask']) * lbls
    loss = reduce_sum(ele_loss)
    true_lbl = true_label(features, labels)

    if mode == ModeKeys.EVAL:
        return EstimatorSpec(mode, loss=loss,  eval_metric_ops={'acc': mean(accuracy(argmax(noise_label(labels),axis=1), argmax(to_one_hot(scores), axis=1)))})
    else:
        optimizer = AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=get_global_step())

        return EstimatorSpec(mode, loss=loss, train_op=train_op)



