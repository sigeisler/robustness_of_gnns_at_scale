import os
import math
from rgnn_at_scale.helper.local import setup_logging, build_configs_and_run
import numpy as np


def run_config_test(expected_values, config_files):
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

    for model_label, expectation in expected_accuracy.items():
        model_results = [r for r in results if r.config["model_params"]["label"] == model_label]

        # the model should always be trained & evaluated for three different seeds
        assert len(model_results) == 3

        test_accuracies = np.array([r.result["accuracy"] for r in model_results])
        test_acc_mean = test_accuracies.mean()
        test_acc_std = test_accuracies.std()
        assert expectation["mean"] - test_acc_mean <= expectation["mean_tol"],\
            (f"The {model_label} model's test accuracy mean is {test_acc_mean:.3} and"
                f" not greater than {expectation['mean']:.3} +- {expectation['mean_tol']:.3}")
        assert test_acc_std - expectation["std"] <= expectation["std_tol"],\
            (f"The {model_label} model's test accuracy standard deviation is {test_acc_std:.3} and"
                f" not smaller than {expectation['std']:.3} +- {expectation['std_tol']:.3}")


def run_global_attack_test(expected_accuracy, config_files):
    results = run_config_test(expected_accuracy, config_files)
    results = [pert_res for r in results for pert_res in r.result["results"]]

    for model_label, epsilons_expactation in expected_accuracy.items():
        model_results = [r for r in results if r["label"] == model_label]
        for eps, expectation in epsilons_expactation.items():
            perturbed_accuracies = np.array([r["accuracy"] for r in model_results if r["epsilon"] == eps])
            pert_acc_mean = perturbed_accuracies.mean()
            pert_acc_std = perturbed_accuracies.std()

            assert pert_acc_mean - expectation["mean"] <= expectation["mean_tol"],\
                (f"The {model_label} model's test accuracy mean for eps = {eps} is {pert_acc_mean:.3} and"
                 f" not smaller then {expectation['mean']:.3} +- {expectation['mean_tol']:.3}")

            assert pert_acc_std - expectation["std"] <= expectation["std_tol"],\
                (f"The {model_label} model's test accuracy standard deviation for eps = {eps} is {pert_acc_std:.3} and"
                    f" not smaller then {expectation['std']:.3} +- {expectation['std_tol']:.3}")


def run_local_attack_test(expected_margin, config_files):
    results = run_config_test(expected_margin, config_files)
    results = [pert_res for r in results for pert_res in r.result["results"]]

    for model_label, epsilons_expactation in expected_margin.items():
        model_results = [r for r in results if r["label"] == model_label]
        for eps, expectation in epsilons_expactation.items():
            margin = np.array([r["margin"] for r in model_results if r["epsilon"] == eps])
            margin_mean = margin.mean()
            margin_std = margin.std()

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
                "mean": 0.825,
                "std": 0.007,
                "mean_tol": 0.005,
                "std_tol": 0.002
            },
            "Soft Medoid GDC (T=0.5)": {
                "mean": 0.815,
                "std": 0.007,
                "mean_tol": 0.005,
                "std_tol": 0.002
            },
            "Soft Median GDC (T=0.5)": {
                "mean": 0.828,
                "std": 0.007,
                "mean_tol": 0.005,
                "std_tol": 0.002
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs', 'train', 'cora.yaml')]

        run_training_test(expected_accuracy, config_files)

    def test_cora_train_lineargcn(self):
        expected_accuracy = {
            "Linear GCN": {
                "mean": 0.82,
                "std": 0.009,
                "mean_tol": 0.005,
                "std_tol": 0.002
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs', 'train', 'cora_linear.yaml')]

        run_training_test(expected_accuracy, config_files)

    def test_cora_train_pprgo(self):
        expected_accuracy = {
            "Vanilla PPRGo": {
                "mean": 0.826,
                "std": 0.006,
                "mean_tol": 0.005,
                "std_tol": 0.001
            },
            "Soft Medoid PPRGo (T=0.5)": {
                "mean": 0.818,
                "std": 0.005,
                "mean_tol": 0.005,
                "std_tol": 0.001
            },
            "Soft Median PPRGo (T=0.5)": {
                "mean": 0.819,
                "std": 0.005,
                "mean_tol": 0.005,
                "std_tol": 0.001
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs', 'train', 'cora_pprgo.yaml')]

        run_training_test(expected_accuracy, config_files)

    def test_cora_attack_direct_prbcd(self):
        expected_accuracy = {
            "Vanilla GCN": {
                0: {
                    "mean": 0.825,
                    "std": 0.007,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.629,
                    "std": 0.008,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
            },
            "Soft Median GDC (T=0.5)": {
                0: {
                    "mean": 0.828,
                    "std": 0.004,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.73,
                    "std": 0.006,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
            }
        }

        config_files = [os.path.join('tests', 'experiment_configs', 'attack_evasion_global_direct', 'cora_prbcd.yaml')]

        run_global_attack_test(expected_accuracy, config_files)

    def test_cora_attack_transfer_prbcd(self):
        expected_accuracy = {
            "Vanilla GCN": {
                0: {
                    "mean": 0.825,
                    "std": 0.007,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.616,
                    "std": 0.008,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
            },
            "Soft Median GDC (T=0.5)": {
                0: {
                    "mean": 0.828,
                    "std": 0.004,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.716,
                    "std": 0.006,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
            },
            "Soft Medoid GDC (T=0.5)": {
                0: {
                    "mean": 0.815,
                    "std": 0.007,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.754,
                    "std": 0.007,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
            },
            "Vanilla PPRGo": {
                0: {
                    "mean": 0.826,
                    "std": 0.006,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.703,
                    "std": 0.018,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
            },
            "Soft Median PPRGo (T=0.5)": {
                0: {
                    "mean": 0.819,
                    "std": 0.005,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.762,
                    "std": 0.004,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
            },
            "Soft Medoid PPRGo (T=0.5)": {
                0: {
                    "mean": 0.818,
                    "std": 0.005,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
                },
                0.1: {
                    "mean": 0.755,
                    "std": 0.003,
                    "mean_tol": 0.005,
                    "std_tol": 0.002
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
                    "mean": -0.743,
                    "std": 0.33,
                    "mean_tol": 0.01,
                    "std_tol": 0.05
                },
            },
            "Vanilla PPRGo": {
                1.0: {
                    "mean": 0.011,
                    "std": 0.27,
                    "mean_tol": 0.01,
                    "std_tol": 0.05
                },
            },
            "Soft Median PPRGo (T=0.5)": {
                1.0: {
                    "mean": 0.16,
                    "std": 0.255,
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
#     testsuit.test_cora_train_lineargcn()
#     testsuit.test_cora_attack_direct_prbcd()
#     testsuit.test_cora_attack_transfer_prbcd()
#     testsuit.test_cora_attack_direct_localprbcd()
