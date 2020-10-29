"""To execute multiple experiments from the command line and print the results as markdown table.
"""
import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

from rgnn.local import setup_logging, build_configs_and_run


parser = argparse.ArgumentParser(
    description='Sparse smoothing results on the pretrained models (with binary attributes).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--config-files', nargs='+', type=str,
                    default=[os.path.join('seml', 'smoothing', 'cora_and_citeseer.yaml')],
                    help='Config YAML files. The script deduplicates the configs, but does not check them.')
parser.add_argument('--kwargs', type=json.loads, default='{}', help='Will overwrite the loaded config')


def _sample_params_to_name(pf_minus_adj: float, pf_plus_adj: float,
                           pf_minus_att: float, pf_plus_att: float, **kwargs) -> str:
    if pf_minus_adj == 0 and pf_plus_adj > 0 and pf_minus_att == 0 and pf_plus_att == 0:
        return 'Add edges'
    elif pf_minus_adj > 0 and pf_plus_adj == 0 and pf_minus_att == 0 and pf_plus_att == 0:
        return 'Del. edges'
    elif pf_minus_adj > 0 and pf_plus_adj > 0 and pf_minus_att == 0 and pf_plus_att == 0:
        return 'Add & del. edges'
    elif pf_minus_adj == 0 and pf_plus_adj == 0 and pf_minus_att > 0 and pf_plus_att > 0:
        return 'Add & del. attr.'
    else:
        return 'UNKNOWN'


def print_results(df_result: pd.DataFrame, decimals=3):
    if len(df_result) == 0:
        return
    df_result.sample_params = df_result.sample_params.apply(lambda params: _sample_params_to_name(**params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_result = df_result.groupby(['dataset', 'sample_params', 'label']).accum_certs.agg([np.mean, stats.sem])
    df_result = df_result.reset_index()
    df_result['accum_certs'] = df_result.apply(
        lambda row: f'{row["mean"]:.{decimals}f} ± {row["sem"]:.{decimals}f}',
        axis=1
    )
    df_result = df_result.sort_values(['dataset', 'sample_params', 'label'])
    df_result = pd.pivot_table(
        df_result,
        index=['dataset', 'label'],
        columns=['sample_params'],
        values='accum_certs',
        aggfunc=lambda x: ''.join(x)
    )

    df_result = df_result.reset_index()
    logging.warning(f'Combined results: \n {df_result.to_markdown(showindex=False)}')


def main(args: argparse.Namespace):
    configs, run = build_configs_and_run(args.config_files, 'experiment_smoothing.py', args.kwargs)

    results = []
    for config in configs:
        df = pd.DataFrame(
            run(config_updates=config).result['results']
        )
        df_config = pd.DataFrame(df.shape[0] * [config])
        results.append(pd.concat((df_config, df), axis=1))

    print_results(pd.concat(results, ignore_index=True))


if __name__ == '__main__':
    setup_logging()

    args = parser.parse_args()
    logging.debug(args)

    main(args)
