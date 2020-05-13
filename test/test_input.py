import pathlib
from tensorflow.python import FixedLenFeature, \
    FixedLenSequenceFeature, int64, float32
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.lib.io.tf_record import tf_record_iterator, TFRecordOptions, TFRecordCompressionType

from src.clean_data import grid, anchors, candidates
from src.input import read_schemata, build_schema, parse_example, input_fn, predict_input_fn

test_dir = pathlib.Path(__file__).parent

def helper_spec(dictionary):
    a = {}
    for feature, value in dictionary.items():
        a[feature] = {'type': str(value.dtype), 'shape': value.shape}
    return a

path = f'{test_dir}/resources/part-00000'

context_schema_path = f'{test_dir}/resources/context_schema'
sequence_schema_path = f'{test_dir}/resources/sequence_schema'


class TestInput(TensorFlowTestCase):
    def test_read_schema(self):
        context_schema_path = f'{test_dir}/resources/context_schema'
        sequence_schema_path = f'{test_dir}/resources/sequence_schema'

        expected_context_schema = {"anchor": {"type": "int64", "shape": [1]},
                                   "anchor_label": {"type": "float64", "shape": [10]},
                                   "anchor_lbl_key": {"type": "int64", "shape": [1]},
                                   "context": {"type": "int64", "shape": [1]},
                                   "context_vec": {"type": "float64", "shape": [2]},
                                   "click_position": {"type": "int64", "shape": [1]},
                                   "reco": {"type": "int64", "shape": [10]},
                                   "seen_click_position": {"type": "int64", "shape": [1]},
                                   "seen_mask": {"type": "float64", "shape": [10]}}

        expected_sequence_schema = {"label": {"type": "float64", "shape": [10]},
                                    "lbl_key": {"type": "int64", "shape": [1]},
                                    "context": {"type": "float64", "shape": [10]}}

        ctx_schema, seq_schema = read_schemata(context_schema_path=context_schema_path,
                                               sequence_schema_path=sequence_schema_path)

        self.assertDictEqual(ctx_schema, expected_context_schema)
        self.assertDictEqual(seq_schema, expected_sequence_schema)

    def test_build_schema(self):
        ctx_schema = {'anchor': {'type': 'int64', 'shape': [1]},
                      'anchor_image': {'type': 'float64', 'shape': [3]},
                      'anchor_label': {'type': 'float64', 'shape': [4]},
                      'anchor_lbl_key': {'type': 'int64', 'shape': [1]},
                      'context': {'type': 'int64', 'shape': [1]},
                      'context_vec': {'type': 'float64', 'shape': [2]},
                      'click_position': {'type': 'int64', 'shape': [1]},
                      'reco': {'type': 'int64', 'shape': [3]},
                      'seen_click_position': {'type': 'int64', 'shape': [1]},
                      'seen_mask': {'type': 'float64', 'shape': [3]}}

        seq_schema = {'image': {'type': 'float64', 'shape': [3]},
                      'label': {'type': 'float64', 'shape': [4]},
                      'lbl_key': {'type': 'int64', 'shape': [1]},
                      'context': {'type': 'float64', 'shape': [3]}}

        expected_tf_ctx_schema = {'anchor': FixedLenFeature(shape=[1], dtype=int64),
                                  'anchor_image': FixedLenFeature(shape=[3], dtype=float32),
                                  'anchor_label': FixedLenFeature(shape=[4], dtype=float32),
                                  'anchor_lbl_key': FixedLenFeature(shape=[1], dtype=int64),
                                  'context': FixedLenFeature(shape=[1], dtype=int64),
                                  'context_vec': FixedLenFeature(shape=[2], dtype=float32),
                                  'click_position': FixedLenFeature(shape=[1], dtype=int64),
                                  'reco': FixedLenFeature(shape=[3], dtype=int64),
                                  'seen_click_position': FixedLenFeature(shape=[1], dtype=int64),
                                  'seen_mask': FixedLenFeature(shape=[3], dtype=float32)}

        expected_tf_seq_schema = {'image': FixedLenSequenceFeature(shape=[3], dtype=float32),
                                  'label': FixedLenSequenceFeature(shape=[4], dtype=float32),
                                  'lbl_key': FixedLenSequenceFeature(shape=[1], dtype=int64),
                                  'context': FixedLenSequenceFeature(shape=[3], dtype=float32)}

        tf_ctx_schema, tf_seq_schema = build_schema(ctx_schema, seq_schema)

        self.assertDictEqual(tf_ctx_schema, expected_tf_ctx_schema)
        self.assertDictEqual(tf_seq_schema, expected_tf_seq_schema)

    def test_parse_example(self):

        iterator = tf_record_iterator(path, options=TFRecordOptions(TFRecordCompressionType.GZIP))

        ctx_schema, seq_schema = read_schemata(context_schema_path=context_schema_path,
                                               sequence_schema_path=sequence_schema_path)

        feature_spec = {'anchor': {'type': 'int64', 'shape': (1,)},
                        'anchor_label': {'type':'float32', 'shape': (10,)},
                        'anchor_lbl_key': {'type': 'int64', 'shape': (1,)},
                        'context': {'type': 'int64', 'shape': (1,)},
                        'context_vec': {'type': 'float32', 'shape': (2,)},
                        'position_vectors': {'type': 'float32', 'shape': (10, 10)},
                        'label': {'type': 'float32', 'shape': (10, 10)},
                        'lbl_key': {'type': 'int64', 'shape': (10, 1)}}
        label_spec = {'reco': {'type': 'int64', 'shape': (10,)},
                      'click_position': {'type': 'int64', 'shape': (1,)},
                      'seen_click_position': {'type': 'int64', 'shape': (1,)},
                      'seen_mask': {'type': 'float32', 'shape': (10,)},
                      'normal_mask': {'type': 'float32', 'shape': (10,)}}

        ctx_schema, seq_schema = build_schema(ctx_schema, seq_schema)

        with self.test_session() as sess:
            context, sequences = parse_example(ctx_schema, seq_schema, next(iterator))
            features, labels = sess.run([context, sequences])
            print(labels)
            context, sequences = parse_example(ctx_schema, seq_schema, next(iterator))
            features, labels = sess.run([context, sequences])
            print(features)
            print(labels)
            print('###############################')
            self.assertDictEqual(helper_spec(features), feature_spec)
            self.assertDictEqual(helper_spec(labels), label_spec)

    def test_input_fn(self):

        ctx_schema, seq_schema = read_schemata(context_schema_path=context_schema_path,
                                               sequence_schema_path=sequence_schema_path)

        features_spec = {'anchor': {'type': 'int64', 'shape': (3,1)},
                         'anchor_label': {'type':'float32', 'shape': (3,10)},
                         'anchor_lbl_key': {'type': 'int64', 'shape': (3,1)},
                         'context': {'type': 'int64', 'shape': (3,1)},
                         'context_vec': {'type': 'float32', 'shape': (3,2)},
                         'position_vectors': {'type': 'float32', 'shape': (3, 10, 10)},
                         'label': {'type': 'float32', 'shape': (3, 10, 10)},
                         'lbl_key': {'type': 'int64', 'shape': (3, 10, 1)}}

        labels_spec = {'reco': {'type': 'int64', 'shape': (3, 10)},
                       'click_position': {'type': 'int64', 'shape': (3, 1)},
                       'seen_click_position': {'type': 'int64', 'shape': (3, 1)},
                       'seen_mask': {'type': 'float32', 'shape': (3, 10)},
                       'normal_mask': {'type': 'float32', 'shape': (3,10)}}


        ctx_schema, seq_schema = build_schema(ctx_schema, seq_schema)

        params = {'batch_size': 3, 'buffer_size': 5, 'epochs': 10, 'cycle_length': 5}

        dataset = input_fn(params=params, sequence_schema=seq_schema, context_schema=ctx_schema, part_files=[path]).make_one_shot_iterator()

        with self.test_session() as sess:
            features, labels = sess.run(dataset.get_next())
            print(labels)
            self.assertDictEqual(helper_spec(features), features_spec)
            self.assertDictEqual(helper_spec(labels), labels_spec)

    def test_predict_input_fn(self):
        expected_features = {'anchor_label': anchors, 'label': candidates}

        features = predict_input_fn({'epochs': 1, 'batch_size': 10}, grid).make_one_shot_iterator()

        with self.test_session() as sess:
            features = sess.run(features.get_next())
            self.assertDictEqual({k:v.tolist() for k,v in features.items()}, expected_features)