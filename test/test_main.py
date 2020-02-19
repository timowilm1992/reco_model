from tensorflow.python import parse_single_sequence_example, FixedLenFeature, \
    FixedLenSequenceFeature, int64, float32
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.lib.io.tf_record import tf_record_iterator, TFRecordOptions, TFRecordCompressionType

from src.input import read_schemata, build_schema, parse_example


class TestMain(TensorFlowTestCase):
    def test_read_schema(self):
        context_schema_path = 'resources/context_schema'
        sequence_schema_path = 'resources/sequence_schema'

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
        path = 'resources/part-00001'
        context_schema_path = 'resources/context_schema'
        sequence_schema_path = 'resources/sequence_schema'

        iterator = tf_record_iterator(path, options=TFRecordOptions(TFRecordCompressionType.GZIP))

        ctx_schema, seq_schema = read_schemata(context_schema_path=context_schema_path,
                                               sequence_schema_path=sequence_schema_path)


        ctx_schema, seq_schema = build_schema(ctx_schema, seq_schema)

        with self.test_session() as sess:
            context, sequences = parse_example(ctx_schema, seq_schema, next(iterator))
            ctx, sequences = sess.run([context, sequences])

            print(ctx)