import argparse
import json
import logging
import os
from glob import glob
from pathlib import Path
from rgnn_at_scale.helper.local import setup_logging, build_configs_and_run


parser = argparse.ArgumentParser(
    description='Execute experiments contained in yaml.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--config-file', type=str,
                    default=os.path.join('config', 'train', 'cora_and_citeseer.yaml'),
                    help='Config YAML files. The script deduplicates the configs, but does not check them.')
parser.add_argument('--kwargs', type=json.loads, default='{}', help='Will overwrite the loaded config')
parser.add_argument('--output', type=str, default=os.path.join('output'),
                    help="Folder to which the sacred experiment results will be logged.")


def main(args: argparse.Namespace):
    configs, run = build_configs_and_run([args.config_file], 'experiment_train.py', args.kwargs)

    for config in configs:
        try:
            result = run(config_updates=config)
        except Exception as e:
            logging.exception(e)
            logging.error(
                f"Failed to run config {config}")
            continue

        try:
            serializable_result = dict(config=result.config,
                                       status=result.status,
                                       start_time=str(result.start_time),
                                       stop_time=str(result.stop_time),
                                       result=result.result)
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            file_prefix = f'{result.experiment_info["name"]}_{result.config["dataset"]}'
            existing_files = glob(str(output_dir / file_prefix) + "_*.json")
            uid = 0
            if len(existing_files) > 0:
                uid = max([int(Path(existing_file).name.split("_")[-1].split(".json")[0])
                           for existing_file in existing_files]) + 1

            filename = f"{file_prefix}_{uid:06d}.json"
            with open(output_dir / filename, "w")as f:
                f.write(json.dumps(serializable_result, indent=4))
        except Exception as e:
            logging.exception(e)
            logging.error(
                f"Failed to save results of run to disk {config}")
            continue


if __name__ == '__main__':
    setup_logging()

    args = parser.parse_args()
    logging.debug(args)

    main(args)
