import json
from functools import partial

from multiprocessing import cpu_count

from tensorflow import FixedLenFeature
from tensorflow import FixedLenSequenceFeature
from tensorflow import decode_json_example
from tensorflow import float32
from tensorflow import int64
from tensorflow import ones
from tensorflow import parse_single_sequence_example
from tensorflow import shape
from tensorflow.contrib.data import parallel_interleave
from tensorflow.contrib.data import shuffle_and_repeat
from tensorflow.core.example.feature_pb2 import FloatList, Feature, Int64List, FeatureList
from tensorflow.python.data import Dataset
from tensorflow.python.data import TFRecordDataset



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
    labels = {}
    context, sequence = parse_single_sequence_example(serialized_example, context_schema, sequence_schema)
    labels['reco'] = context.pop('reco')
    labels['click_position'] = context.pop('click_position')
    labels['seen_click_position'] = context.pop('seen_click_position')
    labels['seen_mask'] = context.pop('seen_mask')
    labels['normal_mask'] = ones([shape(labels['seen_mask'])[0]])
    context['position_vectors'] = sequence['context']
    context['label'] = sequence['label']
    context['lbl_key'] = sequence['lbl_key']
    features = context
    return features,labels

def trans_predict_example(an, can):
    return {'anchor_label': an, 'label': can}


def predict_input_fn(params, data):
    dataset = Dataset.from_tensor_slices(data)
    dataset = dataset.map(trans_predict_example).repeat(params['epochs'])
    dataset = dataset.batch(params['batch_size'])
    return dataset


def input_fn(params, sequence_schema, context_schema, part_files):
    dataset = Dataset.from_tensor_slices(part_files).shuffle(len(part_files))
    dataset = dataset.apply(parallel_interleave(lambda file: TFRecordDataset(file, compression_type='GZIP'),
                                                cycle_length=params['cycle_length'], sloppy=True))
    dataset = dataset.map(partial(parse_example, context_schema, sequence_schema), num_parallel_calls=cpu_count())
    dataset = dataset.apply(shuffle_and_repeat(params['buffer_size'], count = params['epochs']))
    dataset = dataset.batch(params['batch_size'])
    return dataset





