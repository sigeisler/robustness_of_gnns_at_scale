import argparse
import json
import logging
import os

from rgnn.local import setup_logging, build_configs_and_run


parser = argparse.ArgumentParser(
    description='Train models.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--config-files', nargs='+', type=str,
                    default=[os.path.join('seml', 'train', 'cora_and_citeseer.yaml')],
                    help='Config YAML files. The script deduplicates the configs, but does not check them.')
parser.add_argument('--kwargs', type=json.loads, default='{}', help='Will overwrite the loaded config')


def main(args: argparse.Namespace):
    configs, run = build_configs_and_run(args.config_files, 'experiment_train.py', args.kwargs)

    for config in configs:
        run(config_updates=config)


if __name__ == '__main__':
    setup_logging()

    args = parser.parse_args()
    logging.debug(args)

    main(args)
