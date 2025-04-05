import logging
from decimal import Decimal
from enum import Enum
from time import time

import numpy as np
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValue import DiscreteValue
from geniusweb.issuevalue.Domain import Domain
from geniusweb.profile.utilityspace import ValueSetUtilities
from geniusweb.profile.utilityspace.DiscreteValueSetUtilities import DiscreteValueSetUtilities
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from numpy.typing import NDArray
from tudelft_utilities_logging import ReportToLogger

from agents.group44_agent.utils.opponent_model import OpponentModel
from agents.group44_agent.utils.warmup_counter import WarmupCounter


class InitializationMode(Enum):
    UNIFORM = "uniform"
    RANDOM = "random"
    CUSTOM = "custom_profile"
    CONSTANT = "constant"

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


class LinearAdditiveUtilitySpaceOptimizer(OpponentModel, LinearAdditiveUtilitySpace):
    def __init__(self, domain: Domain, name: str, logger: ReportToLogger, warmup_rounds: int = 50, update_rounds: int = 10, learning_rate: float = 0.1, epochs: int = 100,
                 weights_init_mode: InitializationMode = InitializationMode.UNIFORM, values_init_mode: InitializationMode = InitializationMode.RANDOM,
                 init_utility_space: LinearAdditiveUtilitySpace = None, init_constant_value: float = None):
        OpponentModel.__init__(self, domain, name, logger, warmup_rounds=warmup_rounds, update_rounds=update_rounds, learning_rate=learning_rate, epochs=epochs,
                               weights_init_mode=weights_init_mode,
                               values_init_mode=values_init_mode, init_constant_value = init_constant_value)
        self.warmup_rounds = warmup_rounds
        self.update_rounds = update_rounds
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.warmup_counter = WarmupCounter(warmup_rounds)

        self.weights_init_mode = weights_init_mode
        self.values_init_mode = values_init_mode

        self.init_utility_space = init_utility_space
        self.init_constant_value = init_constant_value

        if weights_init_mode == InitializationMode.CUSTOM and init_utility_space is None:
            self.logger.log(logging.WARNING, f"Weights custom initialization mode requires a custom utility space profile! Defaulting weights initialization mode to uniform mode.")
            self.weights_init_mode = InitializationMode.UNIFORM
        if values_init_mode == InitializationMode.CUSTOM and init_utility_space is None:
            self.logger.log(logging.WARNING, f"Values custom initialization mode requires a custom utility space profile! Defaulting values initialization mode to uniform mode.")
            self.values_init_mode = InitializationMode.UNIFORM
        if values_init_mode == InitializationMode.CONSTANT and init_constant_value is None:
            self.logger.log(logging.WARNING, f"Values constant initialization mode requires a constant init value! Defaulting values initialization mode to uniform mode.")
            self.values_init_mode = InitializationMode.UNIFORM

        self.raw_values: NDArray[NDArray[np.float64]] = None
        self.raw_weights: NDArray[np.float64] = None

        self.normalized_values: NDArray[NDArray[np.float64]] = None
        self.normalized_weights: NDArray[np.float64] = None

        self._initialize_issues_raw_weights()
        self._initialize_issues_raw_values()

        self._normalize_issues_weights()
        self._normalize_issues_values()

        issues_weights_dict = self._construct_issues_weights_dict()
        issues_values_dict = self._construct_issues_values_dict()

        LinearAdditiveUtilitySpace.__init__(self, domain, f"{name}s_opponent_predicted_profile", issues_values_dict, issues_weights_dict)

        self.num_samples_statistics_cached = 0

        self.epoch_losses_per_round: list[list[float]] = []
        self.num_samples_per_round: list[int] = []

        # cache
        # bids frequency difference between offered and received bids for each issue and value
        self.bids_frequency_difference: NDArray[NDArray[np.int32]] = np.array([np.zeros(self.domain.getValues(issue).size()) for issue in sorted(self.domain.getIssues())],
                                                                              dtype=object)

    def _initialize_issues_raw_weights(self, enable_warning: bool = True) -> None:
        num_issues = len(self.domain.getIssues())
        if self.weights_init_mode == InitializationMode.UNIFORM:
            # softmax weight normalization assumption
            # processed_weight_i = exp(raw_weight_i)/sum(exp(raw_weight_j))
            # if all raw_weight_j =0 then processed_weight_i = 1/num_issues -> uniform
            self.raw_weights = np.zeros(num_issues, dtype=np.float64)
        elif self.weights_init_mode == InitializationMode.RANDOM:
            # We assume the raw weights are in range [-20, 20] because e^20~2^32
            self.raw_weights = 40.0 * np.random.random(num_issues) - 20
        elif self.weights_init_mode == InitializationMode.CUSTOM:
            # softmax weight normalization assumption
            # processed_weight_i = exp(raw_weight_i)/sum(exp(raw_weight_j))
            # -> log(processed_weight_i) = raw_weight_i - log(sum(exp(raw_weight_j))) -> raw_weight_i = log(processed_weight_i * constant)
            # here we choose the constant to be num_issues
            self.raw_weights = np.log([float(weight) * num_issues + 1e-6 for weight in self.init_utility_space.getWeights().values()])
        else:
            # This case should technically never happen
            if enable_warning:
                self.logger.log(logging.WARNING, f"Initialization mode {self.weights_init_mode} is not implemented for weights! Defaulting to uniform initialization mode.")
            self._initialize_issues_raw_weights(False)

    def _initialize_issues_raw_values(self, enable_warning: bool = True) -> None:
        num_values = np.array([self.domain.getValues(issue).size() for issue in sorted(self.domain.getIssues())])
        if self.values_init_mode == InitializationMode.UNIFORM:
            # sigmoid value normalization assumption
            # processed_value = 1/(1+exp(-raw_value))
            # for uniform processed_value=1/num_values => raw_value = - log(num_values-1)
            self.raw_values = np.array([-np.log((nv - 1 + 1e-6) * np.ones(nv)) for nv in num_values], dtype=object)
        elif self.values_init_mode == InitializationMode.RANDOM:
            # We assume the raw values are in range [-20, 20] because e^20~2^32
            self.raw_values = np.array([40.0 * np.random.random(nv) - 20 for nv in num_values], dtype=object)
        elif self.values_init_mode == InitializationMode.CUSTOM:
            # sigmoid value normalization assumption
            # processed_value = 1/(1+exp(-raw_value))
            # raw_value = -log(1/processed_value - 1)
            self.raw_values = np.array(
                [np.array([-np.log(1 / (float(self.init_utility_space.getUtilities()[issue].getUtility(value)) + 1e-6) - 1 + 1e-6) for value in self.domain.getValues(issue)]) for issue in
                 sorted(self.domain.getIssues())], dtype=object)
        elif self.values_init_mode == InitializationMode.CONSTANT:
            self.raw_values = np.array(
                [np.array([-np.log(1 / (self.init_constant_value + 1e-6) - 1 + 1e-6) for _ in self.domain.getValues(issue)]) for
                 issue in
                 sorted(self.domain.getIssues())], dtype=object)
        else:
            # This case should technically never happen
            if enable_warning:
                self.logger.log(logging.WARNING, f"Initialization mode {self.values_init_mode} is not implemented for values utilities! Defaulting to random initialization mode.")
            self._initialize_issues_raw_values(False)

    def _normalize_issues_weights(self) -> None:
        # softmax weight normalization
        self.normalized_weights = np.exp(np.clip(self.raw_weights, -20.0, 20.0)) / np.sum(np.exp(np.clip(self.raw_weights, -20.0, 20.0)))

    def _normalize_issues_values(self) -> None:
        self.normalized_values = np.array([1 / (1 + np.exp(np.clip(-value, -20.0, 20.0))) for value in self.raw_values], dtype=object)

    def _construct_issues_weights_dict(self) -> dict[str, Decimal]:
        return {issue: Decimal(weight) for issue, weight in zip(sorted(self.domain.getIssues()), self.normalized_weights)}

    def _construct_issues_values_dict(self) -> dict[str, ValueSetUtilities]:
        return {issue: DiscreteValueSetUtilities({DiscreteValue(value.getValue()): Decimal(utility) for value, utility in zip(self.domain.getValues(issue), utilities)}) for
                issue, utilities in zip(sorted(self.domain.getIssues()), self.normalized_values)}

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
            return True
        return False

    def _update_cache(self) -> None:
        for i in range(self.num_samples_statistics_cached, len(self.received_bids)):
            # Update bids_frequency_difference
            for j, issue in enumerate(sorted(self.domain.getIssues())):
                self.bids_frequency_difference[j] += (
                        np.array([1 if self.domain.getValues(issue).get(index) == self.offered_bids[i].getValue(issue) else 0 for index in
                                  range(self.domain.getValues(issue).size())]) - np.array(
                    [1 if self.domain.getValues(issue).get(index) == self.received_bids[i].getValue(issue) else 0 for index in
                     range(self.domain.getValues(issue).size())])
                )
        self.num_samples_statistics_cached = len(self.received_bids)

    def _reset_profile(self):
        self._initialize_issues_raw_weights()
        self._initialize_issues_raw_values()

        self._normalize_issues_weights()
        self._normalize_issues_values()

        issues_weights_dict = self._construct_issues_weights_dict()
        issues_values_dict = self._construct_issues_values_dict()

        self._issueWeights = issues_weights_dict
        self._issueUtilities = issues_values_dict

    def _gradient_descent_update(self) -> None:
        losses = []
        for epoch in range(self.epochs):
            # Compute values gradient
            values_gradient = self.normalized_weights  * self.bids_frequency_difference * self.normalized_values * (1 - self.normalized_values)
            # Compute weights gradient
            fv = np.array([f.dot(v) for f, v in zip(self.bids_frequency_difference, self.normalized_values)])
            weights_gradient = self.normalized_weights   * (- self.normalized_weights.dot(fv) * np.ones(len(self.normalized_weights)) + fv)

            # Update values
            self.raw_values -= self.learning_rate * values_gradient
            # Update weights
            # We divide learning rate by a factor to avoid getting stuck in local minima and disallow weights to converge to the boundaries [0,1]
            # This works because we expect the weights to be almost near the opponents weights so we don't have to change them much
            self.raw_weights -= self.learning_rate/500 * weights_gradient

            self._normalize_issues_values()
            self._normalize_issues_weights()

            issues_weights_dict = self._construct_issues_weights_dict()
            issues_values_dict = self._construct_issues_values_dict()

            self._issueWeights = issues_weights_dict
            self._issueUtilities = issues_values_dict

            # Update losses
            loss = np.mean([float(self.getUtility(self.offered_bids[i]) - self.getUtility(self.received_bids[i])) for i in range(self.num_samples_statistics_cached)])
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
