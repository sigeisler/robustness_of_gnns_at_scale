"""To execute multiple experiments from the command line and print the results as markdown table.
"""
import argparse
import json
import logging

import pandas as pd

from rgnn_at_scale.helper.io import Storage
from rgnn_at_scale.helper.local import setup_logging


parser = argparse.ArgumentParser(
    description='Sparse smoothing results on the pretrained models (with binary attributes).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--artifact_dir', type=str, default='cache',
                    help='Dir where the artifacts (e.g. models) are located at.')
parser.add_argument('--artifact_type', type=str, default='pretrained',
                    help='Type of the artifact (should related to a `<type>.json` file in `artifact_dir`).')
parser.add_argument('--filer_kwargs', type=json.loads, default='{}', help='Further filer kwargs.')
parser.add_argument('--output_csv', type=str, default='./artifacts.csv', help='Output of artifacts.')


def main(args: argparse.Namespace):
    storage = Storage(args.artifact_dir)
    artifacts = storage.find_artifacts(args.artifact_type, match_condition=args.filer_kwargs)

    df_artifacts = pd.DataFrame([
        {'_id': artifact['id'], 'time': artifact['time'], **artifact['params']}
        for artifact in artifacts
    ])
    df_artifacts = df_artifacts.sort_values(
        [col for col in ['dataset', 'label', 'seed'] if col in df_artifacts.columns]
    )
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print()
    df_artifacts.to_csv(args.output_csv)


if __name__ == '__main__':
    setup_logging()

    args = parser.parse_args()
    logging.debug(args)

    main(args)
