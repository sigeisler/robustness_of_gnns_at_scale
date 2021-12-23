import os
import numpy as np
import pandas as pd
from rgnn_at_scale.helper.local import setup_logging, build_configs_and_run
from itertools import groupby
from shutil import rmtree
import torch
# clean cache

if os.path.isdir('cache_test'):
    rmtree('cache_test')


def run_config_test(expected_values, config_files):

    if not torch.cuda.is_available():
        config_files = [config_file.replace(".yaml", "_cpu.yaml") for config_file in config_files]

    configs, run = build_configs_and_run(config_files)
    results = []
    for config in configs:

        result = run(config_updates=config)
        results.append(result)
        assert result.status == "COMPLETED"

        if "model_label" in result.config.keys():
            model_label = result.config["model_label"]
        else:
            model_label = result.config["model_params"]["label"]

        assert model_label in expected_values.keys()

    return results


def run_training_test(expected_accuracy, config_files):
    results = run_config_test(expected_accuracy, config_files)

    test_accuracies = [(r.config["model_params"]["label"], r.result["accuracy"]) for r in results]
    test_accuracies = {label: np.array([v for l, v in value])
                       for (label, value) in groupby(test_accuracies, lambda x: x[0])}

    test_accuracies_mean = {label: acc.mean() for label, acc in test_accuracies.items()}
    test_accuracies_std = {label: acc.std() for label, acc in test_accuracies.items()}

    for model_label, expectation in expected_accuracy.items():
        if model_label in test_accuracies.keys():
            # each model should always be trained & evaluated for three different seeds
            assert len(test_accuracies[model_label]) == 3

            assert expectation["mean"] - test_accuracies_mean[model_label] <= expectation["mean_tol"],\
                (f"The {model_label} model's test accuracy mean is {test_accuracies_mean[model_label]:.3} and"
                    f" not greater than {expectation['mean']:.3} +- {expectation['mean_tol']:.3}")

            assert test_accuracies_std[model_label] - expectation["std"] <= expectation["std_tol"],\
                (f"The {model_label} model's test accuracy standard deviation is {test_accuracies_std[model_label]:.3} and"
                    f" not smaller than {expectation['std']:.3} +- {expectation['std_tol']:.3}")


def run_global_attack_test(expected_accuracy, config_files):
    results = run_config_test(expected_accuracy, config_files)
    results_df = pd.DataFrame([pert_res for r in results for pert_res in r.result["results"]])

    results_stats = results_df.groupby(["label", "epsilon"]).describe()

    for model_label, epsilons_expactation in expected_accuracy.items():
        if model_label in list(results_stats.reset_index()["label"]):
            for eps, expectation in epsilons_expactation.items():
                assert results_stats.loc[(model_label, eps)][("accuracy", "count")] == 3

                pert_acc_mean = results_stats.loc[(model_label, eps)][("accuracy", "mean")]
                pert_acc_std = results_stats.loc[(model_label, eps)][("accuracy", "std")]

                assert pert_acc_mean - expectation["mean"] <= expectation["mean_tol"],\
                    (f"The {model_label} model's test accuracy mean for eps = {eps} is {pert_acc_mean:.3} and"
                     f" not smaller then {expectation['mean']:.3} +- {expectation['mean_tol']:.3}")

                assert pert_acc_std - expectation["std"] <= expectation["std_tol"],\
                    (f"The {model_label} model's test accuracy standard deviation for eps = {eps} is {pert_acc_std:.3} and"
                        f" not smaller then {expectation['std']:.3} +- {expectation['std_tol']:.3}")


def run_local_attack_test(expected_margin, config_files):
    results = run_config_test(expected_margin, config_files)

    results_df = pd.DataFrame([pert_res for r in results for pert_res in r.result["results"]])

    results_stats = results_df.groupby(["label", "epsilon"]).describe()[["margin"]]

    for model_label, epsilons_expactation in expected_margin.items():
        if model_label in list(results_stats.reset_index()["label"]):
            for eps, expectation in epsilons_expactation.items():

                assert results_stats.loc[(model_label, eps)][("margin", "count")] == 3 * 8

                margin_mean = results_stats.loc[(model_label, eps)][("margin", "mean")]
                margin_std = results_stats.loc[(model_label, eps)][("margin", "std")]

                assert margin_mean - expectation["mean"] <= expectation["mean_tol"],\
                    (f"The {model_label} model's test accuracy mean for eps = {eps} is {margin_mean:.3} and"
                     f" not smaller then {expectation['mean']:.3} +- {expectation['mean_tol']:.3}")

                assert margin_std - expectation["std"] <= expectation["std_tol"],\
                    (f"The {model_label} model's test accuracy standard deviation for eps = {eps} is {margin_std:.3} and"
                        f" not smaller then {expectation['std']:.3} +- {expectation['std_tol']:.3}")


class TestExperimentTrain():

    def test_cora_train(self):
        expected_accuracy = {
            "Vanilla GCN": {
                "mean": 0.80,
                "std": 0.003,
                "mean_tol": 0.01,
                "std_tol": 0.002
            },
            "Soft Medoid GDC (T=0.5)": {
                "mean": 0.81,
                "std": 0.06,
                "mean_tol": 0.01,
                "std_tol": 0.005
            },
            "Soft Median GDC (T=0.5)": {
                "mean": 0.82,
                "std": 0.003,
                "mean_tol": 0.01,
                "std_tol": 0.005
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs', 'train', 'cora.yaml')]
        run_training_test(expected_accuracy, config_files)

    def test_cora_train_pprgo(self):
        expected_accuracy = {
            "Vanilla PPRGo": {
                "mean": 0.61,
                "std": 0.06,
                "mean_tol": 0.02,
                "std_tol": 0.02
            },
            "Soft Medoid PPRGo (T=0.5)": {
                "mean": 0.41,
                "std": 0.10,
                "mean_tol": 0.02,
                "std_tol": 0.03
            },
            "Soft Median PPRGo (T=0.5)": {
                "mean": 0.67,
                "std": 0.11,
                "mean_tol": 0.02,
                "std_tol": 0.03
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs', 'train', 'cora_pprgo.yaml')]

        run_training_test(expected_accuracy, config_files)

    def test_cora_attack_direct_prbcd(self):
        expected_accuracy = {
            "Vanilla GCN": {
                0.1: {
                    "mean": 0.68,
                    "std": 0.006,
                    "mean_tol": 0.01,
                    "std_tol": 0.02
                },
            },
            "Soft Median GDC (T=0.5)": {
                0.1: {
                    "mean": 0.74,
                    "std": 0.004,
                    "mean_tol": 0.01,
                    "std_tol": 0.02
                },
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs', 'attack_evasion_global_direct', 'cora_prbcd.yaml')]

        run_global_attack_test(expected_accuracy, config_files)

    def test_cora_attack_direct_greedy_rbcd(self):
        expected_accuracy = {
            "Vanilla GCN": {
                0.1: {
                    "mean": 0.75,
                    "std": 0.02,
                    "mean_tol": 0.01,
                    "std_tol": 0.02
                },
            },
            "Soft Median GDC (T=0.5)": {
                0.1: {
                    "mean": 0.78,
                    "std": 0.014,
                    "mean_tol": 0.01,
                    "std_tol": 0.02
                },
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs',
                                     'attack_evasion_global_direct', 'cora_greedy_rbcd.yaml')]

        run_global_attack_test(expected_accuracy, config_files)

    def test_cora_attack_transfer_prbcd(self):
        expected_accuracy = {
            "Vanilla GCN": {
                0.1: {
                    "mean": 0.68,
                    "std": 0.04,
                    "mean_tol": 0.02,
                    "std_tol": 0.02
                },
            },
            "Soft Median GDC (T=0.5)": {
                0.1: {
                    "mean": 0.73,
                    "std": 0.006,
                    "mean_tol": 0.02,
                    "std_tol": 0.02
                },
            },
            "Soft Medoid GDC (T=0.5)": {
                0.1: {
                    "mean": 0.75,
                    "std": 0.002,
                    "mean_tol": 0.02,
                    "std_tol": 0.02
                },
            },
            "Vanilla PPRGo": {
                0.1: {
                    "mean": 0.52,
                    "std": 0.08,
                    "mean_tol": 0.02,
                    "std_tol": 0.02
                },
            },
            "Soft Median PPRGo (T=0.5)": {
                0.1: {
                    "mean": 0.62,
                    "std": 0.13,
                    "mean_tol": 0.02,
                    "std_tol": 0.02
                },
            },
            "Soft Medoid PPRGo (T=0.5)": {
                0.1: {
                    "mean": 0.38,
                    "std": 0.12,
                    "mean_tol": 0.02,
                    "std_tol": 0.02
                },
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs',
                                     'attack_evasion_global_transfer', 'cora_prbcd.yaml')]

        run_global_attack_test(expected_accuracy, config_files)

    def test_cora_attack_direct_localprbcd(self):
        expected_margin = {
            "Vanilla GCN": {
                1.0: {
                    "mean": -0.22,
                    "std": 0.25,
                    "mean_tol": 0.02,
                    "std_tol": 0.1
                },
            },
            "Vanilla PPRGo": {
                1.0: {
                    "mean": -0.01,
                    "std": 0.02,
                    "mean_tol": 0.01,
                    "std_tol": 0.05
                },
            },
            "Soft Median PPRGo (T=0.5)": {
                1.0: {
                    "mean": 0.17,
                    "std": 0.27,
                    "mean_tol": 0.01,
                    "std_tol": 0.05
                },
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs',
                                     'attack_evasion_local_direct', 'cora_localprbcd.yaml')]

        run_local_attack_test(expected_margin, config_files)


# if __name__ == '__main__':
#     testsuit = TestExperimentTrain()
#     testsuit.test_cora_train()
#     testsuit.test_cora_train_pprgo()
#     #testsuit.test_cora_attack_direct_greedy_rbcd()
#     #testsuit.test_cora_attack_direct_prbcd()
#     #testsuit.test_cora_attack_transfer_prbcd()
#     testsuit.test_cora_attack_direct_localprbcd()
