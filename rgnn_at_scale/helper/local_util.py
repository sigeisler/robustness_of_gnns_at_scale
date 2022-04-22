
import logging
import os
import random
import json
from pathlib import Path
import yaml
import jsonpickle
import ast
import copy
import itertools
from itertools import combinations
import numpy as np


VALID_CONFIG_VALUES = ['executable', 'name', 'output_dir',
                       'conda_environment', 'project_root_dir']
VALID_SLURM_CONFIG_VALUES = ['experiments_per_job', 'max_simultaneous_jobs',
                             'sbatch_options_template', 'sbatch_options']

RESERVED_KEYS = ['grid', 'fixed', 'random']


class InputError(SystemExit):
    """Parent class for input errors that don't print a stack trace."""
    pass


class ConfigError(InputError):
    """Raised when the something is wrong in the config"""

    def __init__(self, message="The config file contains an error."):
        super().__init__(f"CONFIG ERROR: {message}")


class ExecutableError(InputError):
    """Raised when the something is wrong with the Python executable"""

    def __init__(self, message="The Python executable has a problem."):
        super().__init__(f"EXECUTABLE ERROR: {message}")


def restore(flat):
    """
    Restore more complex data that Python's json can't handle (e.g. Numpy arrays).
    Copied from sacred.serializer for performance reasons.
    """
    return jsonpickle.decode(json.dumps(flat), keys=True)


def _convert_value(value):
    """
    Parse string as python literal if possible and fallback to string.
    Copied from sacred.arg_parser for performance reasons.
    """

    try:
        return restore(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        # use as string if nothing else worked
        return value


def convert_values(val):
    if isinstance(val, dict):
        for key, inner_val in val.items():
            val[key] = convert_values(inner_val)
    elif isinstance(val, list):
        for i, inner_val in enumerate(val):
            val[i] = convert_values(inner_val)
    elif isinstance(val, str):
        return _convert_value(val)
    return val


class YamlUniqueLoader(yaml.FullLoader):
    """
    Custom YAML loader that disallows duplicate keys

    From https://github.com/encukou/naucse_render/commit/658197ed142fec2fe31574f1ff24d1ff6d268797
    Workaround for PyYAML issue: https://github.com/yaml/pyyaml/issues/165
    This disables some uses of YAML merge (`<<`)
    """


def set_executable_and_working_dir(config_path, seml_dict):
    """
    Determine the working directory of the project and chdir into the working directory.
    Parameters
    ----------
    config_path: Path to the config file
    seml_dict: seml config dictionary

    Returns
    -------
    None
    """
    config_dir = str(Path(config_path).expanduser().resolve().parent)

    working_dir = config_dir
    os.chdir(working_dir)
    if "executable" not in seml_dict:
        raise ConfigError("Please specify an executable path for the experiment.")
    executable = seml_dict['executable']
    executable_relative_to_config = os.path.exists(executable)
    executable_relative_to_project_root = False
    if 'project_root_dir' in seml_dict:
        working_dir = str(Path(seml_dict['project_root_dir']).expanduser().resolve())
        seml_dict['use_uploaded_sources'] = True
        os.chdir(working_dir)  # use project root as base dir from now on
        executable_relative_to_project_root = os.path.exists(executable)
        del seml_dict['project_root_dir']  # from now on we use only the working dir
    else:
        seml_dict['use_uploaded_sources'] = False
        logging.warning("'project_root_dir' not defined in seml config. Source files will not be saved in MongoDB.")
    seml_dict['working_dir'] = working_dir
    if not (executable_relative_to_config or executable_relative_to_project_root):
        raise ExecutableError("Could not find the executable.")
    executable = str(Path(executable).expanduser().resolve())
    seml_dict['executable'] = (str(Path(executable).relative_to(working_dir)) if executable_relative_to_project_root
                               else str(Path(executable).relative_to(config_dir)))


def merge_dicts(dict1, dict2):
    """Recursively merge two dictionaries.

    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.

    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k] = merge_dicts(dict1[k], dict2[k])
            else:
                return_dict[k] = dict2[k]

    return return_dict


def unflatten(dictionary: dict, sep: str = '.', recursive: bool = False, levels=None):
    """
    Turns a flattened dict into a nested one, e.g. {'a.b':2, 'c':3} becomes {'a':{'b': 2}, 'c': 3}
    From https://stackoverflow.com/questions/6037503/python-unflatten-dict.

    Parameters
    ----------
    dictionary: dict to be un-flattened
    sep: separator with which the nested keys are separated
    recursive: bool, default: False
        Whether to also un-flatten sub-dictionaries recursively. NOTE: if recursive is True, there can be key
        collisions, e.g.: {'a.b': 3, 'a': {'b': 5}}. In these cases, keys which are later in the insertion order
        overwrite former ones, i.e. the example above returns {'a': {'b': 5}}.
    levels: int or list of ints (optional).
        If specified, only un-flatten the desired levels. E.g., if levels= [0, -1], then {'a.b.c.d': 111} becomes
        {'a': {'b.c': {'d': 111}}}.

    Returns
    -------
    result_dict: the nested dictionary.
    """

    duplicate_key_warning_str = ("Duplicate key detected in recursive dictionary unflattening, most likely resulting"
                                 " from combining dot-dict notation with nested dictionaries, e.g. {'a.b': 3, 'a':"
                                 " {'b': 5}}. Overwriting any previous entries, which may be undesired.")

    if levels is not None:
        if not isinstance(levels, tuple) and not isinstance(levels, list):
            levels = [levels]
        if len(levels) == 0:
            raise ValueError("Need at least one level to unflatten when levels != None.")
        if not isinstance(levels[0], int):
            raise TypeError(f"Levels must be list or set of integers, got type {type(levels[0])}.")

    result_dict = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict) and recursive:
            value = unflatten(value, sep=sep, recursive=True, levels=levels)

        parts = key.split(sep)
        if levels is not None:
            key_levels = levels.copy()
            for ix in range(len(key_levels)):
                if key_levels[ix] < 0:
                    new_ix = len(parts) + key_levels[ix] - 1
                    if key_levels[ix] == -1:  # special case so that indexing with -1 never throws an error.
                        new_ix = max(0, new_ix)
                    if new_ix < 0:
                        raise IndexError(f"Dictionary key level out of bounds. ({new_ix} < 0).")
                    key_levels[ix] = new_ix
                if key_levels[ix] >= len(parts):
                    raise IndexError(f"Dictionary key level {key_levels[ix]} out of bounds for size {len(parts)}.")
            key_levels = sorted(key_levels)

            key_levels = list(set(key_levels))
            new_parts = []
            ix_current = 0
            for level in key_levels:
                new_parts.append(sep.join(parts[ix_current:level + 1]))
                ix_current = level + 1

            if ix_current < len(parts):
                new_parts.append(sep.join(parts[ix_current::]))
            parts = new_parts

        d = result_dict
        # Index the existing dictionary in a nested way via the separated key levels. Create empty dicts if necessary.
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            elif not isinstance(d[part], dict):
                # Here we have a case such as: {'a.b': ['not_dict'], 'a': {'b': {'c': 111}}}
                # Since later keys overwrite former ones, we replace the value for {'a.b'} with {'c': 111}.
                logging.warning(duplicate_key_warning_str)
                d[part] = dict()
            # Select the sub-dictionary for the key level.
            d = d[part]
        last_key = parts[-1]
        if last_key in d:
            if isinstance(value, dict):
                intersection = set(d[last_key].keys()).intersection(value.keys())
                if len(intersection) > 0:
                    logging.warning(duplicate_key_warning_str)
                # Merge dictionaries, overwriting any existing values for duplicate keys.
                d[last_key] = merge_dicts(d[last_key], value)
            else:
                logging.warning(duplicate_key_warning_str)
                d[last_key] = value
        else:
            d[last_key] = value
    return result_dict


def flatten(dictionary: dict, parent_key: str = '', sep: str = '.'):
    """
    Flatten a nested dictionary, e.g. {'a':{'b': 2}, 'c': 3} becomes {'a.b':2, 'c':3}.
    From https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Parameters
    ----------
    dictionary: dict to be flattened
    parent_key: string to prepend the key with
    sep: level separator

    Returns
    -------
    flattened dictionary.
    """
    import collections

    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_parameter_collections(input_config: dict):
    flattened_dict = flatten(input_config)
    parameter_collection_keys = [k for k in flattened_dict.keys()
                                 if flattened_dict[k] == "parameter_collection"]
    if len(parameter_collection_keys) > 0:
        logging.warning("Parameter collections are deprecated. Use dot-notation for nested parameters instead.")
    while len(parameter_collection_keys) > 0:
        k = parameter_collection_keys[0]
        del flattened_dict[k]
        # sub1.sub2.type ==> # sub1.sub2
        k = ".".join(k.split(".")[:-1])
        parameter_collections_params = [param_key for param_key in flattened_dict.keys() if param_key.startswith(k)]
        for p in parameter_collections_params:
            if f"{k}.params" in p:
                new_key = p.replace(f"{k}.params", k)
                if new_key in flattened_dict:
                    raise ConfigError(f"Could not convert parameter collections due to key collision: {new_key}.")
                flattened_dict[new_key] = flattened_dict[p]
                del flattened_dict[p]
        parameter_collection_keys = [k for k in flattened_dict.keys()
                                     if flattened_dict[k] == "parameter_collection"]
    return unflatten(flattened_dict)


def unpack_config(config):
    config = convert_parameter_collections(config)
    children = {}
    reserved_dict = {}
    for key, value in config.items():
        if not isinstance(value, dict):
            continue

        if key not in RESERVED_KEYS:
            children[key] = value
        else:
            if key == 'random':
                if 'samples' not in value:
                    raise ConfigError('Random parameters must specify "samples", i.e. the number of random samples.')
                reserved_dict[key] = value
            else:
                reserved_dict[key] = value
    return reserved_dict, children


def standardize_config(config: dict):
    config = unflatten(flatten(config), levels=[0])
    out_dict = {}
    for k in RESERVED_KEYS:
        if k == "fixed":
            out_dict[k] = config.get(k, {})
        else:
            out_dict[k] = unflatten(config.get(k, {}), levels=[-1])
    return out_dict


def invert_config(config: dict):
    reserved_sets = [(k, set(config.get(k, {}).keys())) for k in RESERVED_KEYS]
    inverted_config = {}
    for k, params in reserved_sets:
        for p in params:
            inv = inverted_config.get(p, [])
            inv.append(k)
            inverted_config[p] = inv
    return inverted_config


def detect_duplicate_parameters(inverted_config: dict, sub_config_name: str = None, ignore_keys: dict = None):
    if ignore_keys is None:
        ignore_keys = {'random': ('seed', 'samples')}

    duplicate_keys = []
    for p, l in inverted_config.items():
        if len(l) > 1:
            if 'random' in l and p in ignore_keys['random']:
                continue
            duplicate_keys.append((p, l))

    if len(duplicate_keys) > 0:
        if sub_config_name:
            raise ConfigError(f"Found duplicate keys in sub-config {sub_config_name}: "
                              f"{duplicate_keys}")
        else:
            raise ConfigError(f"Found duplicate keys: {duplicate_keys}")

    start_characters = set([x[0] for x in inverted_config.keys()])
    buckets = {k: {x for x in inverted_config.keys() if x.startswith(k)} for k in start_characters}

    if sub_config_name:
        error_str = (f"Conflicting parameters in sub-config {sub_config_name}, most likely "
                     "due to ambiguous use of dot-notation in the config dict. Found "
                     "parameter '{p1}' in dot-notation starting with other parameter "
                     "'{p2}', which is ambiguous.")
    else:
        error_str = ("Conflicting parameters, most likely "
                     "due to ambiguous use of dot-notation in the config dict. Found "
                     "parameter '{p1}' in dot-notation starting with other parameter "
                     "'{p2}', which is ambiguous.")

    for k in buckets.keys():
        for p1, p2 in combinations(buckets[k], r=2):
            if p1.startswith(f"{p2}."):   # with "." after p2 to catch cases like "test" and "test1", which are valid.
                raise ConfigError(error_str.format(p1=p1, p2=p2))
            elif p2.startswith(f"{p1}."):
                raise ConfigError(error_str.format(p1=p1, p2=p2))


def read_config(config_path):
    with open(config_path, 'r') as conf:
        config_dict = convert_values(yaml.load(conf, Loader=YamlUniqueLoader))

    if "seml" not in config_dict:
        raise ConfigError("Please specify a 'seml' dictionary.")

    seml_dict = config_dict['seml']
    del config_dict['seml']

    for k in seml_dict.keys():
        if k not in VALID_CONFIG_VALUES:
            raise ConfigError(f"{k} is not a valid value in the `seml` config block.")

    set_executable_and_working_dir(config_path, seml_dict)

    if 'output_dir' in seml_dict:
        seml_dict['output_dir'] = str(Path(seml_dict['output_dir']).expanduser().resolve())

    if 'slurm' in config_dict:
        slurm_dict = config_dict['slurm']
        del config_dict['slurm']

        for k in slurm_dict.keys():
            if k not in VALID_SLURM_CONFIG_VALUES:
                raise ConfigError(f"{k} is not a valid value in the `slurm` config block.")

        return seml_dict, slurm_dict, config_dict
    else:
        return seml_dict, None, config_dict


def generate_configs(experiment_config):
    """Generate parameter configurations based on an input configuration.

    Input is a nested configuration where on each level there can be 'fixed', 'grid', and 'random' parameters.

    In essence, we take the cartesian product of all the `grid` parameters and take random samples for the random
    parameters. The nested structure makes it possible to define different parameter spaces e.g. for different datasets.
    Parameter definitions lower in the hierarchy overwrite parameters defined closer to the root.

    For each leaf configuration we take the maximum of all num_samples values on the path since we need to have the same
    number of samples for each random parameter.

    For each configuration of the `grid` parameters we then create `num_samples` configurations of the random
    parameters, i.e. leading to `num_samples * len(grid_configurations)` configurations.

    See Also `examples/example_config.yaml` and the example below.

    Parameters
    ----------
    experiment_config: dict
        Dictionary that specifies the "search space" of parameters that will be enumerated. Should be
        parsed from a YAML file.

    Returns
    -------
    all_configs: list of dicts
        Contains the individual combinations of the parameters.


    """

    reserved, next_level = unpack_config(experiment_config)
    reserved = standardize_config(reserved)
    if not any([len(reserved.get(k, {})) > 0 for k in RESERVED_KEYS]):
        raise ConfigError("No parameters defined under grid, fixed, or random in the config file.")
    level_stack = [('', next_level)]
    config_levels = [reserved]
    final_configs = []

    detect_duplicate_parameters(invert_config(reserved), None)

    while len(level_stack) > 0:
        current_sub_name, sub_vals = level_stack.pop(0)
        sub_config, sub_levels = unpack_config(sub_vals)
        if current_sub_name != '' and not any([len(sub_config.get(k, {})) > 0 for k in RESERVED_KEYS]):
            raise ConfigError(f"No parameters defined under grid, fixed, or random in sub-config {current_sub_name}.")
        sub_config = standardize_config(sub_config)
        config_above = config_levels.pop(0)

        inverted_sub_config = invert_config(sub_config)
        detect_duplicate_parameters(inverted_sub_config, current_sub_name)

        inverted_config_above = invert_config(config_above)
        redefined_parameters = set(inverted_sub_config.keys()).intersection(set(inverted_config_above.keys()))

        if len(redefined_parameters) > 0:
            logging.info(f"Found redefined parameters in sub-config '{current_sub_name}': {redefined_parameters}. "
                         f"Definitions in sub-configs override more general ones.")
            config_above = copy.deepcopy(config_above)
            for p in redefined_parameters:
                sections = inverted_config_above[p]
                for s in sections:
                    del config_above[s][p]

        config = merge_dicts(config_above, sub_config)

        if len(sub_levels) == 0:
            final_configs.append((current_sub_name, config))

        for sub_name, sub_vals in sub_levels.items():
            new_sub_name = f'{current_sub_name}.{sub_name}' if current_sub_name != '' else sub_name
            level_stack.append((new_sub_name, sub_vals))
            config_levels.append(config)

    all_configs = []
    for subconfig_name, conf in final_configs:
        conf = standardize_config(conf)
        random_params = conf['random'] if 'random' in conf else {}
        fixed_params = flatten(conf['fixed']) if 'fixed' in conf else {}
        grid_params = conf['grid'] if 'grid' in conf else {}

        if len(random_params) > 0:
            num_samples = random_params['samples']
            root_seed = random_params.get('seed', None)
            random_sampled = sample_random_configs(flatten(random_params), seed=root_seed, samples=num_samples)

        grids = [generate_grid(v, parent_key=k) for k, v in grid_params.items()]
        grid_configs = dict([sub for item in grids for sub in item])
        grid_product = list(cartesian_product_dict(grid_configs))

        with_fixed = [{**d, **fixed_params} for d in grid_product]
        if len(random_params) > 0:
            with_random = [{**grid, **random} for grid in with_fixed for random in random_sampled]
        else:
            with_random = with_fixed
        all_configs.extend(with_random)

    # Cast NumPy integers to normal integers since PyMongo doesn't like them
    all_configs = [{k: int(v) if isinstance(v, np.integer) else v
                    for k, v in config.items()}
                   for config in all_configs]

    all_configs = [unflatten(conf) for conf in all_configs]
    return all_configs


def sample_random_configs(random_config, samples=1, seed=None):
    """
    Sample random configurations from the specified search space.

    Parameters
    ----------
    random_config: dict
        dict where each key is a parameter and the value defines how the random sample is drawn. The samples will be
        drawn using the function sample_parameter.
    samples: int
        The number of samples to draw per parameter
    seed: int or None
        The seed to use when drawing the parameter value. Defaults to None.

    Returns
    -------
    random_configurations: list of dicts
        List of dicts, where each dict gives a value for all parameters defined in the input random_config dict.

    """

    if len(random_config) == 0:
        return [{}]

    rdm_keys = [k for k in random_config.keys() if k not in ["samples", "seed"]]
    random_config = {k: random_config[k] for k in rdm_keys}
    random_parameter_dicts = unflatten(random_config, levels=-1)
    random_samples = [sample_parameter(random_parameter_dicts[k], samples, seed, parent_key=k)
                      for k in random_parameter_dicts.keys()]
    random_samples = dict([sub for item in random_samples for sub in item])
    random_configurations = [{k: v[ix] for k, v in random_samples.items()} for ix in range(samples)]

    return random_configurations


def sample_parameter(parameter, samples, seed=None, parent_key=''):
    """
    Generate random samples from the specified parameter.

    The parameter types are inspired from https://github.com/hyperopt/hyperopt/wiki/FMin. When implementing new types,
    please make them compatible with the hyperopt nomenclature so that we can switch to hyperopt at some point.

    Parameters
    ----------
    parameter: dict
        Defines the type of parameter. Dict must include the key "type" that defines how the parameter will be sampled.
        Supported types are
            - choice: Randomly samples <samples> entries (with replacement) from the list in parameter['options']
            - uniform: Uniformly samples between 'min' and 'max' as specified in the parameter dict.
            - loguniform:  Uniformly samples in log space between 'min' and 'max' as specified in the parameter dict.
            - randint: Randomly samples integers between 'min' (included) and 'max' (excluded).
    samples: int
        Number of samples to draw for the parameter.
    seed: int
        The seed to use when drawing the parameter value. Defaults to None.
    parent_key: str
        The key to prepend the parameter name with. Used for nested parameters, where we here create a flattened version
        where e.g. {'a': {'b': 11}, 'c': 3} becomes {'a.b': 11, 'c': 3}

    Returns
    -------
    return_items: tuple(str, np.array or list)
        tuple of the parameter name and a 1-D list/array of the samples drawn for the parameter.

    """

    if "type" not in parameter:
        raise ConfigError(f"No type found in parameter {parameter}")
    return_items = []
    allowed_keys = ['seed', 'type']
    if seed is not None:
        np.random.seed(seed)
    elif 'seed' in parameter:
        np.random.seed(parameter['seed'])

    param_type = parameter['type']

    if param_type == "choice":
        choices = parameter['options']
        allowed_keys.append("options")
        sampled_values = [random.choice(choices) for _ in range(samples)]
        return_items.append((parent_key, sampled_values))

    elif param_type == "uniform":
        min_val = parameter['min']
        max_val = parameter['max']
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.random.uniform(min_val, max_val, samples)
        return_items.append((parent_key, sampled_values))

    elif param_type == "loguniform":
        if parameter['min'] <= 0:
            raise ConfigError("Cannot take log of values <= 0")
        min_val = np.log(parameter['min'])
        max_val = np.log(parameter['max'])
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.exp(np.random.uniform(min_val, max_val, samples))
        return_items.append((parent_key, sampled_values))

    elif param_type == "randint":
        min_val = int(parameter['min'])
        max_val = int(parameter['max'])
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.random.randint(min_val, max_val, samples)
        return_items.append((parent_key, sampled_values))

    elif param_type == "randint_unique":
        min_val = int(parameter['min'])
        max_val = int(parameter['max'])
        allowed_keys.extend(['min', 'max'])
        sampled_values = np.random.choice(np.arange(min_val, max_val), samples, replace=False)
        return_items.append((parent_key, sampled_values))

    elif param_type == "parameter_collection":
        sub_items = [sample_parameter(v, parent_key=f'{parent_key}.{k}',
                                      seed=seed, samples=samples) for k, v in parameter['params'].items()]
        return_items.extend([sub_item for item in sub_items for sub_item in item])

    else:
        raise ConfigError(f"Parameter type {param_type} not implemented.")

    if param_type != "parameter_collection":
        extra_keys = set(parameter.keys()).difference(set(allowed_keys))
        if len(extra_keys) > 0:
            raise ConfigError(f"Unexpected keys in parameter definition. Allowed keys for type '{param_type}' are "
                              f"{allowed_keys}. Unexpected keys: {extra_keys}")
    return return_items


def generate_grid(parameter, parent_key=''):
    """
    Generate a grid of parameter values from the input configuration.

    Parameters
    ----------
    parameter: dict
        Defines the type of parameter. Options for parameter['type'] are
            - choice: Expects a list of options in paramter['options'], which will be returned.
            - range: Expects 'min', 'max', and 'step' keys with values in the dict that are used as
                     np.arange(min, max, step)
            - uniform: Generates the grid using np.linspace(min, max, num, endpoint=True)
            - loguniform: Uniformly samples 'num' points in log space (base 10) between 'min' and 'max'
            - parameter_collection: wrapper around a dictionary of parameters (of the types above); we call this
              function recursively on each of the sub-parameters.
    parent_key: str
        The key to prepend the parameter name with. Used for nested parameters, where we here create a flattened version
        where e.g. {'a': {'b': 11}, 'c': 3} becomes {'a.b': 11, 'c': 3}

    Returns
    -------
    return_items: tuple(str, list)
        Name of the parameter and list containing the grid values for this parameter.

    """
    if "type" not in parameter:
        raise ConfigError(f"No type found in parameter {parameter}")

    param_type = parameter['type']
    allowed_keys = ['type']

    return_items = []

    if param_type == "choice":
        values = parameter['options']
        allowed_keys.append('options')
        return_items.append((parent_key, values))

    elif param_type == "range":
        min_val = parameter['min']
        max_val = parameter['max']
        step = int(parameter['step'])
        allowed_keys.extend(['min', 'max', 'step'])
        values = list(np.arange(min_val, max_val, step))
        return_items.append((parent_key, values))

    elif param_type == "uniform":
        min_val = parameter['min']
        max_val = parameter['max']
        num = int(parameter['num'])
        allowed_keys.extend(['min', 'max', 'num'])
        values = list(np.linspace(min_val, max_val, num, endpoint=True))
        return_items.append((parent_key, values))

    elif param_type == "loguniform":
        min_val = parameter['min']
        max_val = parameter['max']
        num = int(parameter['num'])
        allowed_keys.extend(['min', 'max', 'num'])
        values = np.logspace(np.log10(min_val), np.log10(max_val), num, endpoint=True)
        return_items.append((parent_key, values))

    elif param_type == "parameter_collection":
        sub_items = [generate_grid(v, parent_key=f'{parent_key}.{k}') for k, v in parameter['params'].items()]
        return_items.extend([sub_item for item in sub_items for sub_item in item])

    else:
        raise ConfigError(f"Parameter {param_type} not implemented.")

    if param_type != "parameter_collection":
        extra_keys = set(parameter.keys()).difference(set(allowed_keys))
        if len(extra_keys) > 0:
            raise ConfigError(f"Unexpected keys in parameter definition. Allowed keys for type '{param_type}' are "
                              f"{allowed_keys}. Unexpected keys: {extra_keys}")

    return return_items


def cartesian_product_dict(input_dict):
    """Compute the Cartesian product of the input dictionary values.
    Parameters
    ----------
    input_dict: dict of lists

    Returns
    -------
    list of dicts
        Cartesian product of the lists in the input dictionary.

    """

    keys = input_dict.keys()
    vals = input_dict.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
