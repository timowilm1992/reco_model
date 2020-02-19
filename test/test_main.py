import os

from tensorflow.python.framework.test_util import TensorFlowTestCase

from src.main import read_schemata


class TestMain(TensorFlowTestCase):

    def test_read_schema(self):
        context_schema_path = 'resources/context_schema'
        sequence_schema_path = 'resources/sequence_schema'

        expected_context_schema = {"anchor": {"type": "int64", "shape": [1]}, "anchor_label": {"type": "float64", "shape": [10]}, "anchor_lbl_key": {"type": "int64", "shape": [1]}, "context": {"type": "int64", "shape": [1]}, "context_vec": {"type": "float64", "shape": [2]}, "click_position": {"type": "int64", "shape": [1]}, "reco": {"type": "int64", "shape": [10]}, "seen_click_position": {"type": "int64", "shape": [1]}, "seen_mask": {"type": "float64", "shape": [10]}}

        expected_sequence_schema = {"label": {"type": "float64", "shape": [10]}, "lbl_key": {"type": "int64", "shape": [1]}, "context": {"type": "float64", "shape": [10]}}

        ctx_schema, seq_schema = read_schemata(context_schema_path=context_schema_path, sequence_schema_path=sequence_schema_path)

        self.assertDictEqual(ctx_schema, expected_context_schema)
        self.assertDictEqual(seq_schema, expected_sequence_schema)

