import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class StatisticalEvaluator:
    # train_fn(seed) -> model, eval_fn(model) -> dict of metrics
    # Returns dict with keys: 'model_name', 'n_runs', and one key per metric
    # Each metric dict has: 'mean', 'std', 'median', 'q25', 'q75', 'n', 'ci_lower', 'ci_upper',
    # 'bootstrap_ci_lower', 'bootstrap_ci_upper', 'values'

    def __init__(self,
                 n_runs: int = 20,
                 confidence_level: float = 0.95,
                 seed_base: int = 14):
        self.n_runs = n_runs
        self.confidence_level = confidence_level
        self.seed_base = seed_base
        self.alpha = 1.0 - confidence_level

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _compute_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro')

        if y_proba is not None:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 1:
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            else:
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                except:
                    metrics['auc'] = np.nan
        else:
            metrics['auc'] = np.nan

        return metrics

    def run_evaluation(self,
                       train_fn: Callable,
                       eval_fn: Callable,
                       model_name: str = "Model",
                       verbose: bool = True) -> Dict:
        all_metrics = defaultdict(list)

        for run_idx in range(self.n_runs):
            seed = self.seed_base + run_idx
            self._set_seed(seed)

            if verbose and (run_idx + 1) % 5 == 0:
                print(f"Run {run_idx + 1}/{self.n_runs}")

            try:
                model = train_fn(seed)
                metrics = eval_fn(model)
                for key, value in metrics.items():
                    if not np.isnan(value):
                        all_metrics[key].append(value)
            except Exception as e:
                if verbose:
                    print(f"Run {run_idx + 1} failed: {e}")
                continue

        results = self._compute_statistics(all_metrics)
        results['model_name'] = model_name
        results['n_runs'] = len(all_metrics.get('auc', []))
        return results

    def _compute_statistics(self, all_metrics: Dict[str, List[float]]) -> Dict:
        results = {}

        for metric_name, values in all_metrics.items():
            if len(values) == 0:
                continue

            values = np.array(values)
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            median = np.median(values)
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)

            se = std / np.sqrt(n)
            t_crit = stats.t.ppf(1 - self.alpha / 2, df=n - 1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se

            bootstrap_ci = self._bootstrap_ci(values, n_bootstrap=10000)

            results[metric_name] = {
                'mean': float(mean),
                'std': float(std),
                'median': float(median),
                'q25': float(q25),
                'q75': float(q75),
                'n': n,
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'bootstrap_ci_lower': float(bootstrap_ci[0]),
                'bootstrap_ci_upper': float(bootstrap_ci[1]),
                'values': values.tolist()
            }

        return results

    def _bootstrap_ci(self, values: np.ndarray, n_bootstrap: int = 10000) -> Tuple[float, float]:
        n = len(values)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        lower = np.percentile(bootstrap_means, 100 * self.alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - self.alpha / 2))
        return (float(lower), float(upper))

    def compare_models(self,
                       results1: Dict,
                       results2: Dict,
                       metric_name: str = 'auc',
                       test_type: str = 'both') -> Dict:
        # Returns dict with keys: 't_test', 'mann_whitney', 'effect_size', 'summary'
        if metric_name not in results1 or metric_name not in results2:
            raise ValueError(f"Metric {metric_name} not found in both results")

        values1 = np.array(results1[metric_name]['values'])
        values2 = np.array(results2[metric_name]['values'])
        comparison = {}

        if test_type in ['parametric', 'both']:
            t_stat, p_value_t = stats.ttest_rel(values1, values2) if len(values1) == len(values2) else stats.ttest_ind(
                values1, values2)
            comparison['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(p_value_t)
            }

        if test_type in ['nonparametric', 'both']:
            u_stat, p_value_mw = stats.mannwhitneyu(values1, values2, alternative='two-sided')
            comparison['mann_whitney'] = {
                'statistic': float(u_stat),
                'p_value': float(p_value_mw)
            }

        mean_diff = np.mean(values1) - np.mean(values2)
        pooled_std = np.sqrt((np.var(values1, ddof=1) + np.var(values2, ddof=1)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        comparison['effect_size'] = {
            'mean_difference': float(mean_diff),
            'cohens_d': float(cohens_d)
        }

        comparison['summary'] = {
            'model1_mean': float(np.mean(values1)),
            'model2_mean': float(np.mean(values2)),
            'model1_std': float(np.std(values1, ddof=1)),
            'model2_std': float(np.std(values2, ddof=1))
        }

        return comparison

    def print_results(self, results: Dict, metric_priority: Optional[List[str]] = None):
        if metric_priority is None:
            metric_priority = ['auc', 'accuracy', 'f1']

        print(f"{results.get('model_name', 'Model')}: {results.get('n_runs', 0)} runs")
        for metric in metric_priority:
            if metric in results:
                m = results[metric]
                print(f"{metric}: {m['mean']:.4f} ± {m['std']:.4f} [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")

    def print_comparison(self, comparison: Dict, metric_name: str = 'auc'):
        summary = comparison['summary']
        effect = comparison['effect_size']
        print(
            f"{metric_name}: {summary['model1_mean']:.4f} vs {summary['model2_mean']:.4f}, diff={effect['mean_difference']:.4f}, d={effect['cohens_d']:.4f}")
        if 't_test' in comparison:
            print(f"p={comparison['t_test']['p_value']:.6f}")


def evaluate_model_predictions(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    evaluator = StatisticalEvaluator(n_runs=1)
    return evaluator._compute_metrics(y_true, y_pred, y_proba)