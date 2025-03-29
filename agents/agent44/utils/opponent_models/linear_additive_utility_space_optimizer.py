import logging
from decimal import Decimal
from enum import Enum

import numpy as np
from geniusweb.issuevalue import Value
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValue import DiscreteValue
from geniusweb.issuevalue.Domain import Domain
from geniusweb.profile.utilityspace.DiscreteValueSetUtilities import DiscreteValueSetUtilities
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profile.utilityspace.ValueSetUtilities import ValueSetUtilities
from numpy.typing import NDArray
from tudelft_utilities_logging import ReportToLogger

from agents.agent44.utils.opponent_model import OpponentModel
from agents.agent44.utils.warmup_counter import WarmupCounter


class InitializationMode(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    CUSTOM = "custom_profile"

    # TODO: Add a frequency based mode to start the weights based on bids frequencies

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return self._value

    @classmethod
    def from_string(cls, str_mode: str):
        if str_mode == cls.UNIFORM.value:
            return cls.UNIFORM
        elif str_mode == cls.RANDOM.value:
            return cls.RANDOM
        elif str_mode == cls.CUSTOM.value:
            return cls.CUSTOM
        else:
            raise ValueError(f"Invalid initialization mode: {str_mode}")


class NormalizationMode(Enum):
    MAX_MIN = "max_min"
    CLIP = "clip"

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return self._value

    @classmethod
    def from_string(cls, str_mode: str):
        if str_mode == cls.MAX_MIN.value:
            return cls.MAX_MIN
        elif str_mode == cls.CLIP.value:
            return cls.CLIP
        else:
            raise ValueError(f"Invalid initialization mode: {str_mode}")


class LinearAdditiveUtilitySpaceOptimizer(OpponentModel, LinearAdditiveUtilitySpace):
    def __init__(self, domain: Domain, name: str, logger: ReportToLogger, warmup_rounds: int = 50, update_rounds: int = 10, learning_rate: float = 0.1, epochs: int = 100,
                 weights_init_mode: InitializationMode = InitializationMode.UNIFORM,
                 weights_norm_mode: NormalizationMode = NormalizationMode.MAX_MIN, values_init_mode: InitializationMode = InitializationMode.RANDOM,
                 values_norm_mode: NormalizationMode = NormalizationMode.CLIP,
                 init_utility_space: LinearAdditiveUtilitySpace = None):
        OpponentModel.__init__(self, domain, name, logger, warmup_rounds=warmup_rounds, update_rounds=update_rounds, learning_rate=learning_rate, epochs=epochs,
                               weights_init_mode=weights_init_mode,
                               weights_normalization_mode=weights_norm_mode,
                               values_init_mode=values_init_mode, values_normalization_mode=values_norm_mode)
        self.warmup_rounds = warmup_rounds
        self.update_rounds = update_rounds
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.warmup_counter = WarmupCounter(warmup_rounds)

        self.weights_init_mode = weights_init_mode
        self.values_init_mode = values_init_mode

        self.init_utility_space = init_utility_space

        if weights_init_mode == InitializationMode.CUSTOM and init_utility_space is None:
            self.logger.log(logging.WARNING, f"Weights custom initialization mode requires a custom utility space profile! Defaulting weights initialization mode to uniform mode.")
            self.weights_init_mode = InitializationMode.UNIFORM
        if values_init_mode == InitializationMode.CUSTOM and init_utility_space is None:
            self.logger.log(logging.WARNING, f"Values custom initialization mode requires a custom utility space profile! Defaulting values initialization mode to uniform mode.")
            self.values_init_mode = InitializationMode.UNIFORM

        issues_weights = self._initialize_issues_weights()
        issues_values_utilities = self._initialize_issues_values_utilities()
        LinearAdditiveUtilitySpace.__init__(self, domain, f"{name}s_opponent_predicted_profile", issues_values_utilities, issues_weights)

        self.weights_norm_mode = weights_norm_mode
        self.values_norm_mode = values_norm_mode

        self.num_samples_statistics_cached = 0

        self.epoch_losses_per_round: list[list[float]] = []
        self.num_samples_per_round: list[int] = []
        # cache
        # bids Sum of One Hot Encoded Difference (SOHED) for each issue
        self.bids_SOHED: dict[str, NDArray[object]] = {issue: np.array([Decimal(0) for _ in range(self.domain.getValues(issue).size())]) for issue in self.domain.getIssues()}

    def _initialize_issues_weights(self, enable_warning: bool = True) -> dict[str, Decimal]:
        num_issues = len(self.domain.getIssues())
        if self.weights_init_mode == InitializationMode.UNIFORM:
            return {issue: Decimal(1 / num_issues) for issue in self.domain.getIssues()}
        elif self.weights_init_mode == InitializationMode.RANDOM:
            weights = np.random.rand(num_issues)
            sum_weights = weights.sum()
            return {issue: Decimal(weights[i] / sum_weights) for i, issue in enumerate(self.domain.getIssues())}
        elif self.weights_init_mode == InitializationMode.CUSTOM:
            return self.init_utility_space.getWeights()
        else:
            # This case should technically never happen
            if enable_warning:
                self.logger.log(logging.WARNING, f"Initialization mode {self.weights_init_mode} is not implemented for weights! Defaulting to uniform initialization mode.")
            return self._initialize_issues_weights()

    def _initialize_issues_values_utilities(self, enable_warning: bool = True) -> dict[
        str, ValueSetUtilities]:
        num_values = {issue: self.domain.getValues(issue).size() for issue in self.domain.getIssues()}
        if self.values_init_mode == InitializationMode.UNIFORM:
            return {issue: DiscreteValueSetUtilities(
                {DiscreteValue(value.getValue()): Decimal(1 / num_values[issue]) for value in self.domain.getValues(issue)}) for issue in self.domain.getIssues()}
        elif self.values_init_mode == InitializationMode.RANDOM:
            return {issue: DiscreteValueSetUtilities(
                {DiscreteValue(value.getValue()): Decimal(np.random.random()) for value in self.domain.getValues(issue)}) for issue in self.domain.getIssues()}
        elif self.values_init_mode == InitializationMode.CUSTOM:
            return self.init_utility_space.getUtilities()
        else:
            # This case should technically never happen
            if enable_warning:
                self.logger.log(logging.WARNING, f"Initialization mode {self.values_init_mode} is not implemented for values utilities! Defaulting to random initialization mode.")
            return self._initialize_issues_values_utilities()

    def get_predicted_utility(self, bid: Bid) -> Decimal:
        return self.getUtility(bid)

    def _update(self) -> None:
        assert (len(self.received_bids) == len(self.offered_bids))

        self._update_cache()
        self.warmup_counter.update()
        if self.warmup_counter.is_warmed_up():
            # Before updating the weights and values reset them to initial state
            self._reset_profile()

            # Use gradient descent to iteratively update the values and weights
            self._gradient_descent_update()

            # Reset counter
            self.warmup_counter = WarmupCounter(self.update_rounds)
            self.warmup_counter.reset()

    def _update_cache(self) -> None:
        for i in range(self.num_samples_statistics_cached, len(self.received_bids)):
            # Update SOHED
            for issue in self.domain.getIssues():
                self.bids_SOHED[issue] += (
                        np.array([Decimal(1) if self.domain.getValues(issue).get(index) == self.offered_bids[i].getValue(issue) else Decimal(0) for index in
                                  range(self.domain.getValues(issue).size())]) - np.array(
                    [Decimal(1) if self.domain.getValues(issue).get(index) == self.received_bids[i].getValue(issue) else Decimal(0) for index in
                     range(self.domain.getValues(issue).size())])
                )
        self.num_samples_statistics_cached = len(self.received_bids)

    def _reset_profile(self):
        self._issueWeights = self._initialize_issues_weights(enable_warning=False)
        self._issueUtilities = self._initialize_issues_values_utilities(enable_warning=False)

    def _normalize_weights(self, denormalized_weights: dict[str, Decimal]) -> dict[str, Decimal]:
        if self.weights_norm_mode == NormalizationMode.MAX_MIN:
            min_weights = min([denormalized_weights[issue] for issue in denormalized_weights])
            max_weights = max([denormalized_weights[issue] for issue in denormalized_weights])
            if min_weights == max_weights:
                return {issue: Decimal(1 / len(denormalized_weights)) for issue in denormalized_weights}
            else:
                scaled_weights = {issue: (denormalized_weights[issue] - min_weights) / (max_weights - min_weights) for issue in denormalized_weights}
                sum_scaled_weights = sum(scaled_weights.values())
                return {issue: scaled_weights[issue] / sum_scaled_weights for issue in denormalized_weights}

        elif self.weights_norm_mode == NormalizationMode.CLIP:
            clipped_weights = {issue: np.clip(float(denormalized_weights[issue]), a_min=0, a_max=1) for issue in denormalized_weights}
            sum_clipped_weights = sum(clipped_weights.values())
            return {issue: Decimal(clipped_weights[issue] / sum_clipped_weights) for issue in denormalized_weights}
        else:
            # Technically we should never reach here
            raise ValueError(f"{self.weights_norm_mode} normalization mode is not implemented in normalize_weights method!")

    def _normalize_values(self, denormalized_values: dict[str, dict[Value, Decimal]]) -> dict[str, dict[Value, Decimal]]:
        if self.values_norm_mode == NormalizationMode.MAX_MIN:
            min_values = {issue: min(denormalized_values[issue].values()) for issue in denormalized_values}
            max_values = {issue: max(denormalized_values[issue].values()) for issue in denormalized_values}

            return {issue: (
                {value: (denormalized_values[issue][value] - min_values[issue]) / (max_values[issue] - min_values[issue]) for value in denormalized_values[issue]}
                if max_values[issue] != min_values[issue] else {value: Decimal(np.clip(float(denormalized_values[issue][value]), a_min=0, a_max=1)) for value in
                                                                denormalized_values[issue]}) for issue in denormalized_values}
        elif self.values_norm_mode == NormalizationMode.CLIP:
            return {issue: {value: Decimal(np.clip(float(denormalized_values[issue][value]), a_min=0, a_max=1)) for value in
                            denormalized_values[issue]} for issue in denormalized_values}
        else:
            # Technically we should never reach here
            raise ValueError(f"{self.values_norm_mode} normalization mode is not implemented in normalize_values method!")

    def _gradient_descent_update(self) -> None:
        losses = []
        for epoch in range(self.epochs):
            # Compute values gradient
            values_gradient = {issue: self._issueWeights[issue] * self.bids_SOHED[issue] / self.num_samples_statistics_cached for issue in self.domain.getIssues()}
            # Compute weights gradient
            weights_gradient = {issue: sum(
                [self.bids_SOHED[issue][index] * self._issueUtilities[issue].getUtility(self.domain.getValues(issue).get(index)) for index in
                 range(self.domain.getValues(issue).size())]) / self.num_samples_statistics_cached for issue in self.domain.getIssues()}

            denormalized_values: dict[str, dict[Value, Decimal]] = {}
            denormalized_weights: dict[str, Decimal] = {}
            # Update
            for issue in self.domain.getIssues():
                # Update values
                denormalized_values[issue] = {DiscreteValue(self.domain.getValues(issue).get(index).getValue()): self._issueUtilities[issue].getUtility(
                    self.domain.getValues(issue).get(index)) - Decimal(self.learning_rate) * values_gradient[issue][index] for index in range(self.domain.getValues(issue).size())}

                # Update weights
                denormalized_weights[issue] = self._issueWeights[issue] - Decimal(self.learning_rate) * weights_gradient[issue]

            # Normalize values
            normalized_values = self._normalize_values(denormalized_values)
            self._issueUtilities = {issue: DiscreteValueSetUtilities(normalized_values[issue]) for issue in normalized_values}

            # Normalize weights
            normalized_weights = self._normalize_weights(denormalized_weights)
            self._issueWeights = normalized_weights

            # Update losses
            loss = sum([self._issueWeights[issue] * sum(
                [self.bids_SOHED[issue][index] * self._issueUtilities[issue].getUtility(self.domain.getValues(issue).get(index)) for index in
                 range(self.domain.getValues(issue).size())]) for issue in self.domain.getIssues()]) / self.num_samples_statistics_cached
            losses.append(loss)
        self.num_samples_per_round.append(len(self.offered_bids))
        self.epoch_losses_per_round.append(losses)

    def get_utility_space(self) -> UtilitySpace:
        return LinearAdditiveUtilitySpace(self.domain, self.name, self._issueUtilities, self._issueWeights)

    def get_utility_space_json_dict(self) -> dict:
        return {
            "LinearAdditiveUtilitySpace": {
                "issueUtilities": {
                    issue: {
                        "DiscreteValueSetUtilities": {
                            "valueUtilities": {
                                value.getValue(): np.round(float(self._issueUtilities[issue].getUtility(value)), 6)
                                for value in self.domain.getValues(issue)
                            }
                        }
                    }
                    for issue in self.domain.getIssues()
                },
                "issueWeights": {
                    issue: np.round(float(self._issueWeights[issue]), 6)
                    for issue in self.domain.getIssues()
                },
                "domain": {
                    "name": self.domain.getName(),
                    "issueValues": {
                        issue: {
                            "values": list([self.domain.getValues(issue).get(index).getValue() for index in range(self.domain.getValues(issue).size())])
                        }
                        for issue in self.domain.getIssues()
                    }
                },
                "name": self.opponent_name + "sPredictedProfile"
            }
        }

    def get_epoch_losses_json_dict(self) -> dict:
        return {
            f"round{i}": {
                "numberOfSamples/numberOfOfferPairs": self.num_samples_per_round[i],
                "epochLosses": list([np.round(float(loss), 6) for loss in self.epoch_losses_per_round[i]])
            }
            for i in range(len(self.epoch_losses_per_round))
        }
