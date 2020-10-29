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
    description='Attack the pretrained models.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--config-files', nargs='+', type=str, default=[os.path.join('seml', 'attack', 'attack.yaml')],
                    help='Config YAML files. The script deduplicates the configs, but does not check them.')
parser.add_argument('--kwargs', type=json.loads, default='{}', help='Will overwrite the loaded config')


def print_results(df_result: pd.DataFrame, decimals=3):
    if len(df_result) == 0:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_result = df_result.groupby(['dataset', 'attack', 'label', 'epsilon']).accuracy.agg([np.mean, stats.sem])
        df_result = df_result.reset_index()
        df_result['accuracy'] = df_result.apply(
            lambda row: f'{row["mean"]:.{decimals}f} ± {row["sem"]:.{decimals}f}',
            axis=1
        )
        df_result = df_result.sort_values(['dataset', 'attack', 'label'])
        df_result = pd.pivot_table(
            df_result,
            index=['dataset', 'label'],
            columns=['attack', 'epsilon'],
            values='accuracy',
            aggfunc=lambda x: ''.join(x)
        )

    df_result = df_result.reset_index()
    headers = [f'{attack} {f"- {epsilon}" if str(epsilon) else ""}' for attack, epsilon in df_result.columns.tolist()]
    logging.warning(f'Combined results: \n {df_result.to_markdown(headers=headers, showindex=False)}')


def main(args: argparse.Namespace):
    configs, run = build_configs_and_run(args.config_files, 'experiment_attack.py', args.kwargs)

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
