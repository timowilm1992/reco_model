import glob


import logging
from tensorflow.python.estimator.training import train_and_evaluate, TrainSpec, EvalSpec

import src.model as model
from functools import partial

from src.clean_data import grid
from src.input import read_schemata, build_schema, predict_input_fn, input_fn

from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator

from src.transform import pairwise_probability, build_diags, optimize

dummi_version = 1
session_packer_version = 1
einpacker_version = 1
base_path = '/home/timo/code/examples/data/'

model_dir = f'{base_path}/model_dir'

dummi_path = 'dummi-mnist-%05d' % dummi_version
session_packer_path = 'session-packer-%05d' % session_packer_version
input_path = f'{base_path}/{dummi_path}/{session_packer_path}'


params = {'buffer_size': 2000, 'batch_size': 16 , 'cycle_length': 4, 'learning_rate': 1e-3, 'epochs': 2}

part = '/home/timo/examples/trainer/test/resources/part-r-00000'


def main():
    sequence_schema_path = f'{input_path}/train/sequence_schema'
    context_schema_path = f'{input_path}/train/context_schema'

    context_schema,  sequence_schema = read_schemata(context_schema_path, sequence_schema_path)

    tf_ctx_schema, tf_seq_schema = build_schema(context_schema, sequence_schema)

    train_parts = glob.glob(input_path + '/train' + '/part-*')
    validation_parts = glob.glob(input_path + '/test' + '/part-*')

    run_config = RunConfig(log_step_count_steps=10,
                           save_checkpoints_steps=100,
                           save_summary_steps=200,
                           keep_checkpoint_max=32)

    shared_input_fn = partial(input_fn, params, tf_seq_schema, tf_ctx_schema)

    train_input_fn = partial(shared_input_fn, train_parts)

    validation_input_fn = partial(shared_input_fn, validation_parts)

    train_spec = TrainSpec(train_input_fn, max_steps=1000000)

    eval_spec = EvalSpec(validation_input_fn, steps=200, name='validation', start_delay_secs=30, throttle_secs=1)

    estimator = Estimator(model_fn=model.model_fn,
                          model_dir=model_dir,
                          params=params,
                          config=run_config)

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.getLogger('tensorflow').propagate = False

    train_and_evaluate(estimator=estimator,
                       train_spec=train_spec,
                       eval_spec=eval_spec)

    prediction = list(estimator.predict(input_fn=partial(predict_input_fn, {'epochs': 1, 'batch_size': 10}, grid)))

    scores = [p.tolist() for p in prediction]

    pairwise_prob = pairwise_probability(scores)

    zero = pairwise_prob[0]

    A_zero = build_diags(zero)

    print(optimize(A_zero).x)




if __name__ == '__main__':
    main()
