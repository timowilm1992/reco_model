import pathlib
from tensorflow.python.ops.math_ops import range, cast

from tensorflow.python import constant, shape, expand_dims, SparseTensor, int64, global_variables_initializer, equal, \
    float32, gather_nd
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.keras.backend import concatenate

from src.model import lookup_positives, create_diffs, create_label, elementwise_loss, predict_scores, true_label

test_dir = pathlib.Path(__file__).parent



context_schema_path = f'{test_dir}/resources/context_schema'
sequence_schema_path = f'{test_dir}/resources/sequence_schema'


class TestMain(TensorFlowTestCase):

    def test_lookup_positives(self):
        scores = constant([[1., 0., 2.],
                           [0., 1., 1.]])
        clicks = constant([[2], [1]])

        expected = [2., 1.]


        positive_scores = lookup_positives(scores, clicks)

        with self.test_session():
            self.assertListEqual(positive_scores.eval().tolist(), expected)

    def test_create_diffs(self):
        scores = constant([[1., 0., 2.],
                           [0., 1., 1.]])
        positive_scores = [2., 1.]

        expected = [[1., 2., 0.], [1., 0., 0.]]

        diffs = create_diffs(positive_scores, scores)

        with self.test_session():
            self.assertListEqual(diffs.eval().tolist(), expected)

    def test_create_label(self):
        click_position = [[1], [3], [0]]
        labels = create_label(click_position,4)
        expected = [[1., 0., 1., 1.],
                    [1., 1., 1., 0.],
                    [0., 1., 1., 1.]]

        with self.test_session():
            self.assertListEqual(labels.eval().tolist(), expected)

    def test_elementwise_loss(self):
        labels = [[0., 1., 0.],
                  [1., 0., 0.]]

        logits = [[1., 0., 0.],
                  [0., 0., 1.]]

        mask = [[1., 1., 0.],
                [0., 1., 1.]]

        expected = [[1.3132616875182228, 0.6931471805599453, 0.],
                    [0., 0.6931471805599453, 1.3132616875182228]]

        with self.test_session():
            self.assertAllClose(elementwise_loss(labels, logits, mask).eval(), expected)

    def test_predict_scores(self):
        features = {'anchor_label': constant([[0., 1], [1., 0.]]), 'label': constant([[[0., 1.], [0., 1.], [0., 1.]], [[1., 0.], [1., 0.], [1., 0.]]])}
        p = predict_scores(features)
        with self.test_session() as sess:
            sess.run(global_variables_initializer())
            print(p.eval())

    def test_true_label(self):
        labels = {'reco':  [[0, 1, 2], [2, 1, 0]]}
        features = {'anchor': [[1], [0]]}
        expected = [[0., 1., 0.],
                    [0., 0., 1.]]
        true_lbl  = true_label(features, labels)
        with self.test_session():
            self.assertListEqual(true_lbl.eval().tolist(), expected)

    def test_ficken(self):
        labels = {'click_position': [1,2], 'reco':[[0, 1, 2], [2, 1, 0]]}
        id = range(shape(labels['click_position'])[0])
        idx = concatenate([expand_dims(cast(id, int64), axis=1), expand_dims(cast(labels['click_position'], int64),axis=1)], axis=1)
        clicked_item = gather_nd(labels['reco'],idx)
        with self.test_session():
            print(clicked_item.eval())
