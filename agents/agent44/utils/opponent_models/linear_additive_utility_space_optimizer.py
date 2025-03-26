from decimal import Decimal
from enum import Enum
from typing import Dict

import numpy as np
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.DiscreteValue import DiscreteValue
from geniusweb.issuevalue.Domain import Domain
from geniusweb.profile.utilityspace.DiscreteValueSetUtilities import DiscreteValueSetUtilities
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profile.utilityspace.ValueSetUtilities import ValueSetUtilities

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
    def __init__(self, domain: Domain, name: str, warmup_rounds: int = 50, learning_rate: float = 0.1,
                 epochs: int = 100,
                 weights_init_mode: str = "uniform", values_init_mode: str = "random",
                 init_utility_space: UtilitySpace = None):
        OpponentModel.__init__(self, domain, warmup_rounds=warmup_rounds, learning_rate=learning_rate, epochs=epochs,
                               weights_init_mode=weights_init_mode, values_init_mode=values_init_mode)
        self.warmup_rounds = warmup_rounds
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.warmup_counter = WarmupCounter(warmup_rounds)

        weights_initialization_mode = self._parse_initialization_mode(weights_init_mode, init_utility_space)
        values_initialization_mode = self._parse_initialization_mode(values_init_mode, init_utility_space)
        issues_weights = self._initialize_issues_weights(weights_initialization_mode, init_utility_space)
        issues_values_utilities = self._initialize_issues_values_utilities(values_initialization_mode,
                                                                           init_utility_space)
        LinearAdditiveUtilitySpace.__init__(self, domain, f"{name}_opponent_predicted_profile", issues_values_utilities,
                                            issues_weights)

        self.num_samples_statistics_cached = 0
        self.lambda_ = 0.5

        # cache
        # bids Sum of One Hot Encoded Difference (SOHED) for each issue
        self.bids_SOHED: Dict[str, np.array] = {issue: np.zeros(domain.getValues(issue).size()) for issue in
                                                domain.getIssues()}

    @staticmethod
    def _parse_initialization_mode(init_mode: str,
                                   init_utility_space: LinearAdditiveUtilitySpace = None) -> InitializationMode:
        try:
            initialization_mode = InitializationMode.from_string(init_mode)
            if initialization_mode == InitializationMode.CUSTOM and init_utility_space is None:
                raise ValueError(f"Custom initialization mode requires a custom utility space profile!")
        except ValueError as e:
            print(e.__str__() + ". Defaulting initialization mode to uniform mode.")
            initialization_mode = InitializationMode.UNIFORM
        return initialization_mode

    def _initialize_issues_weights(self, mode: InitializationMode = InitializationMode.UNIFORM,
                                   init_utility_space: LinearAdditiveUtilitySpace = None) -> Dict[str, Decimal]:
        num_issues = len(self.domain.getIssues())
        if mode == InitializationMode.UNIFORM:
            return {issue: Decimal(1 / num_issues) for issue in self.domain.getIssues()}
        elif mode == InitializationMode.RANDOM:
            weights = np.random.rand(num_issues)
            sum_weights = weights.sum()
            return {issue: Decimal(weights[i] / sum_weights) for i, issue in enumerate(self.domain.getIssues())}
        elif mode == InitializationMode.CUSTOM:
            return init_utility_space.getWeights()
        else:
            # This case should technically never happen
            print(
                f"Initialization mode {mode} is not implemented for weights! Defaulting to uniform initialization mode.")
            return self._initialize_issues_weights(InitializationMode.UNIFORM, init_utility_space)

    def _initialize_issues_values_utilities(self, mode: InitializationMode = InitializationMode.UNIFORM,
                                            init_utility_space: LinearAdditiveUtilitySpace = None) -> Dict[
        str, ValueSetUtilities]:
        num_values = {issue: self.domain.getValues(issue).size() for issue in self.domain.getIssues()}
        if mode == InitializationMode.UNIFORM:
            return {issue: DiscreteValueSetUtilities(
                {DiscreteValue(value.getValue()): Decimal(1 / num_values[issue]) for value in
                 self.domain.getValues(issue)}) for
                issue in self.domain.getIssues()}
        elif mode == InitializationMode.RANDOM:
            return {issue: DiscreteValueSetUtilities(
                {DiscreteValue(value.getValue()): Decimal(np.random.random()) for value in
                 self.domain.getValues(issue)}) for issue
                in self.domain.getIssues()}
        elif mode == InitializationMode.CUSTOM:
            return init_utility_space.getUtilities()
        else:
            # This case should technically never happen
            print(
                f"Initialization mode {mode} is not implemented for values utilities! Defaulting to random initialization mode.")
            return self._initialize_issues_values_utilities(InitializationMode.RANDOM, init_utility_space)

    def get_predicted_utility(self, bid: Bid) -> Decimal:
        return self.getUtility(bid)

    def _update(self) -> None:
        assert (len(self.received_bids) == len(self.offered_bids))

        self._update_cache()
        if self.warmup_counter.is_warmed_up():
            # Use gradient descent to iteratively update the values and weights
            self._gradient_descent_update()

        self.warmup_counter.update()

    def _update_cache(self):
        for i in range(self.num_samples_statistics_cached, len(self.received_bids)):
            # Update SOHED
            for issue in self.domain.getIssues():
                self.bids_SOHED[issue] += np.array(
                    np.array(self.domain.getValues(issue)) == self.offered_bids[i].getValue(
                        issue), dtype=int) - np.array(
                    np.array(self.domain.getValues(issue)) == self.received_bids[i].getValue(issue), dtype=int)
        self.num_samples_statistics_cached = len(self.received_bids)

    def _gradient_descent_update(self):
        for epoch in range(self.epochs):
            # Compute values gradient
            values_gradient = {issue: self._issueWeights[issue] * self.bids_SOHED[issue] for issue in
                               self.domain.getIssues()}
            # Compute weights gradient
            weights_gradient = {issue: sum(
                [self.bids_SOHED[issue][index] * self._issueUtilities[issue].getUtility(self.domain.getValues(issue).get(index)) for index in
                 range(self.domain.getValues(issue).size())]) - 2 * self.lambda_*(1-sum(self._issueWeights.values())) for issue in self.domain.getIssues()}

            # Compute lambdas gradient
            lambdas_gradient= (1-sum(self._issueWeights.values()))**2

            # Update


            # Normalize weights
            pass
