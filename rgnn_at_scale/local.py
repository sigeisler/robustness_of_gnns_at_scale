"""Helper for local execution.
"""

import importlib
import logging
import os
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sacred import Experiment
from seml.config import read_config, generate_configs


def build_configs_and_run(config_files: Sequence[str], executable: Optional[str] = None,
                          kwargs: Dict[str, Any] = {}) -> Tuple[List[Dict[str, Any]], Callable]:
    """Returns all (deduplicated) configs provided in `config_files` and provides the `run`. You can pass the
    config via the `config_updates` argument (see Example below).

    Parameters
    ----------
    config_files : Sequence[str]
        Config (`.yaml`) files of same experiment (all must refer to the same potentially provided executable).
    executable : str, optional
        Optionally the name of the executable, by default None.
    kwargs : Dict[str, Any], optional
        Overwrite/add certain configs (please make sure they are valid!), by default {}.

    Returns
    -------
    Tuple[List[Dict[str, Any]], Callable]
        Configs and the callable of type `sacred.Experiment#run` (pass config via `config_updates` argument).

    Raises
    ------
    ValueError
        If the configs contain multiple executables or the executable has no `sacred.Experiment` attribute.

    Examples
    --------
    >>> configs, run = build_configs_and_run(['a.yaml', 'b.yaml'])
    >>> results = []
    >>> for config in configs:
    >>>     results.append(run(config_updates=config).result)
    """
    configs = []
    executable = None
    for config_file in config_files:
        seml_config, _, experiment_config = read_config(config_file)
        if executable is None:
            executable = seml_config['executable']
        elif executable != seml_config['executable']:
            raise ValueError(
                f'All configs must be for the same executable! Found {executable} and {seml_config["executable"]}.'
            )
        configs.extend(generate_configs(experiment_config))

    # Overwrite/add configs
    for key, value in kwargs.items():
        for config in configs:
            config[key] = value

    deduplicate_index = {
        json.dumps(config, sort_keys=True): i
        for i, config
        in enumerate(configs)
    }
    configs = [configs[i] for i in deduplicate_index.values()]

    module = importlib.import_module(os.path.splitext(os.path.basename(executable))[0])

    run = None
    for attr in dir(module):
        if isinstance(getattr(module, attr), Experiment):
            run = getattr(module, attr).run
    if run is None:
        raise ValueError(f'Executable {executable} has not attribute of type `sacred.Experiment`!')
    return configs, run


def setup_logging():
    """Setup logging for standard out
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
