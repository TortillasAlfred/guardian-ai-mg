# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.metrics import confusion_matrix

from fairlearn.postprocessing._interpolated_thresholder import InterpolatedThresholder
from fairlearn.postprocessing._tradeoff_curve_utilities import (
    METRIC_DICT,
    _extend_confusion_matrix,
    _interpolate_curve,
    _tradeoff_curve,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing._threshold_optimizer import _reformat_and_group_data


class ThresholdOptimizerIncludingLevellingDown(ThresholdOptimizer):
    """
    This postprocessor extends over `ThresholdOptimizer` by taking
    levelling down into account.

    The original algorithm works by grid-searching over
    base rates, fixing them in expectation for every group
    for an expected perfect parity, and then selecting the
    base rate with the maximum accuracy. This version uses
    the same base rate computation at first, but also computes
    levelling down alonside accuracy for every base rate. Instead
    of returning a single classifier, it returns the Pareto
    Frontier of optimal classifiers over the accuracy-levelling down
    tradeoff while keeping the expected disparity to 0.

    Currently assumes that the base rate used in the fairnes metric
    is to be maximized. For instance, TPR disparity is admissible
    but FPR disparity needs to be replaced with TNR disparity to be
    admissible.

    Not yet implemented for Equalized Odds constraint, although
    should pose no conceptual problem.
    """

    def _threshold_optimization_for_simple_constraints(
        self, sensitive_features, labels, scores
    ):
        """ """
        n = len(labels)
        self._tradeoff_curve = {}
        self._x_grid = np.linspace(0, 1, self.grid_size + 1)
        overall_tradeoff_curve = 0 * self._x_grid
        overall_levelling_down_curve = 0 * self._x_grid

        data_grouped_by_sensitive_feature = _reformat_and_group_data(
            sensitive_features, labels, scores
        )

        for (
            sensitive_feature_value,
            group,
        ) in data_grouped_by_sensitive_feature:
            # Determine probability of current sensitive feature group based on data.
            p_sensitive_feature_value = len(group) / n

            roc_convex_hull = _tradeoff_curve(
                group,
                sensitive_feature_value,
                flip=self.flip,
                x_metric=self.x_metric_,
                y_metric=self.y_metric_,
            )

            self._tradeoff_curve[sensitive_feature_value] = _interpolate_curve(
                roc_convex_hull, "x", "y", "operation", self._x_grid
            )

            # Add up objective for the current group multiplied by the probability of the current
            # group. This will help us in identifying the maximum overall objective.
            overall_tradeoff_curve += (
                p_sensitive_feature_value
                * self._tradeoff_curve[sensitive_feature_value]["y"]
            )

            # While overall_tradeoff_curve represents the accuracy over the entire population,
            # overall_levelling_down_curve will represent LD_f, which can be similarly computed
            # by averaging over the levelling down of every group.
            y_true = group["label"]
            y_pred_base = (group["score"] >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_base).ravel()
            base_counts = _extend_confusion_matrix(
                false_positives=fp,
                true_positives=tp,
                true_negatives=tn,
                false_negatives=fn,
            )
            group_base_rate = METRIC_DICT[self.x_metric_](base_counts)
            group_levelling_down = np.maximum(
                group_base_rate - self._tradeoff_curve[sensitive_feature_value]["x"],
                0,
            )
            overall_levelling_down_curve += (
                1 / len(data_grouped_by_sensitive_feature) * group_levelling_down
            )

        acc_ld_tradeoff = pd.DataFrame(
            {
                "accuracy": -overall_tradeoff_curve,  # Use -accuracy because the PF code below assumes costs to be minimized
                "levelling_down": overall_levelling_down_curve,
            }
        )
        acc_ld_tradeoff = acc_ld_tradeoff.to_numpy()

        # Compute Pareto Frontier
        is_efficient = np.ones(acc_ld_tradeoff.shape[0], dtype=bool)
        for i, c in enumerate(acc_ld_tradeoff):
            is_efficient[i] = np.all(
                np.any(acc_ld_tradeoff[:i] > c, axis=1)
            ) and np.all(np.any(acc_ld_tradeoff[i + 1 :] > c, axis=1))

        # For every point in the ParetoFrontier, collect the InterpolatedClassifier
        pareto_frontier_idxs = is_efficient.nonzero()[0]
        pareto_frontier = []
        for i_best in pareto_frontier_idxs:
            # Create the solution as interpolation of multiple points with a separate
            # interpolation per sensitive feature value.
            interpolation_dict = {}
            for sensitive_feature_value in self._tradeoff_curve.keys():
                best_interpolation = self._tradeoff_curve[
                    sensitive_feature_value
                ].transpose()[i_best]
                interpolation_dict[sensitive_feature_value] = Bunch(
                    p0=best_interpolation.p0,
                    operation0=best_interpolation.operation0,
                    p1=best_interpolation.p1,
                    operation1=best_interpolation.operation1,
                )

            classifier = InterpolatedThresholder(
                self.estimator_,
                interpolation_dict,
                prefit=True,
                predict_method=self._predict_method,
            ).fit(None, None)

            pareto_frontier.append(
                {
                    "accuracy": overall_tradeoff_curve[i_best],
                    "levelling_down": overall_levelling_down_curve[i_best],
                    "classifier": classifier,
                    "expected_disparity": 0.0,
                }
            )

        self.interpolated_thresholder_pareto_frontier = pareto_frontier

        return pareto_frontier
