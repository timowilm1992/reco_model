import json

from tensorflow import FixedLenFeature
from tensorflow import FixedLenSequenceFeature
from tensorflow import float32
from tensorflow import int64
from tensorflow import parse_single_sequence_example
from tensorflow.core.example.feature_pb2 import FloatList, Feature, Int64List, FeatureList

def _float_feature(value):
    return Feature(float_list=FloatList(value=value))


def _int_feature(value):
    return Feature(int64_list=Int64List(value=value))


def _to_feature_list(value, trans_fn):
    return FeatureList(feature=[trans_fn(v) for v in value])


schema_type = {'int64': int64, 'float64': float32}

def build_schema(context_schema, sequence_schema):
    tf_context_schema = {feature: FixedLenFeature(shape=attributes['shape'], dtype=schema_type[attributes['type']]) for
                         feature, attributes in context_schema.items()}
    tf_seq_schema = {feature: FixedLenSequenceFeature(shape=attributes['shape'], dtype=schema_type[attributes['type']])
                     for feature, attributes in sequence_schema.items()}
    return tf_context_schema, tf_seq_schema



def read_single_schema(path):
    with open(path) as file:
        return json.loads(file.readline())

def read_schemata(context_schema_path, sequence_schema_path):
    context_schema = read_single_schema(context_schema_path)
    sequence_schema = read_single_schema(sequence_schema_path)
    return context_schema, sequence_schema


def parse_example(context_schema, sequence_schema, serialized_example):
    context, sequence = parse_single_sequence_example(serialized_example, context_schema, sequence_schema)
    return context, sequence










