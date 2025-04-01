import json
import logging
import os
from pathlib import Path
from time import time
from typing import cast

import numpy as np
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from numpy.typing import NDArray
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel
from .utils.opponent_models.linear_additive_utility_space_optimizer import LinearAdditiveUtilitySpaceOptimizer, InitializationMode
from .utils.warmup_counter import WarmupCounter


class Agent44(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        # Extra variables added

        self.opponent_model_warmup_rounds: int = 250  # Number of rounds to not accept any offers with utility less than the best bid before conceding other offers. Each round in opponent model means two rounds here one round of offering a bid and one round of receiving a bid.
        self.opponent_model_update_rounds: int = 100  # Number of rounds to wait once the opponent model is warmed up before updating opponent model every time. Each round in opponent model means two rounds here one round of offering a bid and one round of receiving a bid. Set to -1 to not update the opponent model after the first time
        self.opponent_model_learning_rate: float = 0.05  # Learning rate for updating the opponent model
        self.opponent_model_epochs: int = 10
        # It makes sense to assume the weight that has high value in my profile will likely have high weights in the opponent as well therefore it's best to use my profile issues weights as the starting issues weights
        self.opponent_model_weight_initialization_mode: InitializationMode = InitializationMode.CUSTOM  # Initialization mode for the weights of issues in the opponent utility space model
        self.opponent_model_value_initialization_mode: InitializationMode = InitializationMode.CONSTANT  # Initialization mode for the values of issues in the opponent utility space model
        self.opponent_model_init_constant_value: float = 0.1
        self.opponent_model: OpponentModel = None

        # compose a list of all possible bids
        self.all_bids: AllBidsList = None
        self.last_offered_bid: Bid = None

        # Number of rounds to not accept any offers with utility less than the best bid before conceding other offers
        self.warmup_counter = WarmupCounter(self.opponent_model_warmup_rounds)
        # Percentage of samples of bid space to take when finding a bid before warming up
        # If set to 1 all the bids in all_bids will be considered
        self.percentage_samples: float = 0.1
        self.my_top_bids_utility_pair: NDArray[(Bid, float)] = None
        self.bid_scoring_metric: str = "nash_product"
        self.bid_scoring_is_time_dependant: bool = True
        self.selfishness: float = 0.6  # Selfishness factor for computing wellfare score or nash product score of a bid based on my utility and opponents estimated utility
        self.time_pressure_factor: float = 1  # Time pressure factor for computing wellfare score or nash product score of a bid based on my utility and opponents estimated utility
        self.percentage_top_bid_candidates: float = 0.005  # Percentage of top score bid candidates to sample from when finding a bid
        self.num_top_bid_candidates: int = None  # This will be inferred based on the self.percentage_top_bid_candidates.
        self.min_utility_to_accept: float = None
        self.reorder_bid_scoring = False
        self.repr_args = {"selfishness": self.selfishness, "percentage_bid_samples": self.percentage_samples, "bid_scoring_metric": self.bid_scoring_metric,
                          "bid_scoring_is_time_dependant": self.bid_scoring_is_time_dependant, "time_pressure_factor": self.time_pressure_factor,}

        self.opponent_best_bid: Bid = None
        self.top_bids_score_pair: NDArray[(Bid, float)] = None

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """
        # t0 = time()
        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = cast(LinearAdditiveUtilitySpace, profile_connection.getProfile())
            self.domain = self.profile.getDomain()
            profile_connection.close()

            self.opponent_model = LinearAdditiveUtilitySpaceOptimizer(self.domain, self.me.getName(), self.logger,
                                                                      warmup_rounds=self.opponent_model_warmup_rounds,
                                                                      update_rounds=self.opponent_model_update_rounds,
                                                                      learning_rate=self.opponent_model_learning_rate,
                                                                      epochs=self.opponent_model_epochs,
                                                                      weights_init_mode=self.opponent_model_weight_initialization_mode,
                                                                      values_init_mode=self.opponent_model_value_initialization_mode,
                                                                      init_utility_space=None if self.opponent_model_weight_initialization_mode != InitializationMode.CUSTOM and self.opponent_model_value_initialization_mode != InitializationMode.CUSTOM else self.profile,
                                                                      init_constant_value=self.opponent_model_init_constant_value)
            self.repr_args['opponent_model'] = self.opponent_model.get_repr_json()
            # Make a list of all bids
            self.all_bids = AllBidsList(self.domain)

            # Store the top self.num_top_bid_candidates bids
            self.num_top_bid_candidates = int(self.percentage_top_bid_candidates * self.all_bids.size())
            self.my_top_bids_utility_pair = np.array([(self.all_bids.get(index), utility) for utility, index in
                                                      sorted([(float(self.profile.getUtility(bid)), i) for i, bid in enumerate(self.all_bids)], key=lambda x: x[0], reverse=True)[
                                                      :int(self.percentage_samples * self.all_bids.size())]])
            self.min_utility_to_accept = self.my_top_bids_utility_pair[-1][1]
            print("min utility to accept is ", str(self.min_utility_to_accept))
            self.top_bids_score_pair = np.array(
                sorted([(bid, self.score_bid(bid, our_utility=float(utility))) for bid, utility in self.my_top_bids_utility_pair], key=lambda x: x[1], reverse=True))

            self.repr_args["percentage_samples/num_samples"] = f"{self.percentage_samples}/{len(self.my_top_bids_utility_pair)}"
            self.repr_args["percentage_top_bid_candidates/num_top_bid_candidates"] = f"{self.percentage_top_bid_candidates}/{self.num_top_bid_candidates}"
        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]
                self.opponent_model.set_opponent_name(self.other)

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be sent if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """MUST BE IMPLEMENTED
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities({"SAOP"}, {"geniusweb.profile.utilityspace.LinearAdditive"}, )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Template agent for the ANL 2022 competition"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            bid = cast(Offer, action).getBid()

            # set bid as last received
            self.last_received_bid = bid

            # update opponent model with last received and offered bid
            if self.last_offered_bid is not None:
                # if not self.warmup_counter.is_warmed_up():
                self.reorder_bid_scoring = self.opponent_model.update(self.last_received_bid, self.last_offered_bid)
                self.warmup_counter.update()

            if self.opponent_best_bid is None:
                self.opponent_best_bid = bid
            elif self.profile.getUtility(bid) > self.profile.getUtility(self.opponent_best_bid):
                self.opponent_best_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            self.last_offered_bid = bid
            action = Offer(self.me, bid)

        # self.warmup_counter.update()
        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        if self.other is None:
            self.logger.log(logging.WARNING, "Opponent name was not set; skipping saving opponents predicted profile.")
        else:
            profile_data = json.dumps(self.opponent_model.get_utility_space_json_dict(), sort_keys=True, indent=4)
            with open(Path(f"{self.storage_dir}").joinpath(f"{self.other}_predicted_profile.json"), "w") as f:
                f.write(profile_data)

            if isinstance(self.opponent_model, LinearAdditiveUtilitySpaceOptimizer):
                loss_data = json.dumps(self.opponent_model.get_epoch_losses_json_dict(), sort_keys=True, indent=4)
                with open(Path(f"{self.storage_dir}").joinpath(f"{self.other}_rounds_loss.json"), "w") as f:
                    f.write(loss_data)

                repr_data = json.dumps(self.repr_args, sort_keys=True, indent=4)
                with open(Path("results").joinpath(sorted(os.listdir("results"))[-1]).joinpath(f"agent44_settings.json"), "w") as f:
                    f.write(repr_data)

            self.logger.log(logging.INFO, "Saved opponents predicted profile: " + self.other)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None or not self.warmup_counter.is_warmed_up():
            return False

        if self.last_offered_bid is not None and self.profile.getUtility(bid)>=self.profile.getUtility(self.last_offered_bid):
            return True
        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # very basic approach that accepts if the offer is valued above min utility and
        # 95% of the time towards the deadline has passed
        conditions = [
            self.profile.getUtility(bid) > self.min_utility_to_accept*(1-(progress-0.95)/0.05),
            progress > 0.95,
        ]
        return all(conditions)

    def find_bid(self) -> Bid:
        # t0 = time()
        self.counter += 1
        if self.warmup_counter.is_warmed_up():
            # opponent model is warmed up so we should reorder the top bids based on interpolated score which is based on our and opponents utility using self.score_bid
            if self.reorder_bid_scoring:
                self.logger.log(logging.INFO, "Reordering bids scores based on opponent model")
                self.top_bids_score_pair = np.array(
                    sorted([(bid, self.score_bid(bid, our_utility=float(self.profile.getUtility(bid)))) for bid in self.all_bids], key=lambda x: x[1], reverse=True))
            found_bid = self.top_bids_score_pair[np.random.randint(0, self.num_top_bid_candidates)][0]

        else:
            # Select a random best bid from top self.num_top_bid_candidates bids
            found_bid = self.my_top_bids_utility_pair[np.random.randint(0, self.num_top_bid_candidates)][0]
        # If opponent offered a better bid before offer that back again
        if self.opponent_best_bid is not None and self.profile.getUtility(found_bid) <= self.profile.getUtility(self.opponent_best_bid):
            return self.opponent_best_bid
        else:
            return found_bid

    def score_bid(self, bid: Bid, our_utility: float = None) -> float:
        our_utility = float(self.profile.getUtility(bid)) if our_utility is None else our_utility

        if self.bid_scoring_metric == "social_welfare":
            if self.warmup_counter.is_warmed_up():
                opponent_utility = float(self.opponent_model.get_predicted_utility(bid))
            else:
                opponent_utility = 0.0
            if self.bid_scoring_is_time_dependant:
                time_pressure = self._get_time_pressure()
                score = self.selfishness * our_utility + (1.0 - self.selfishness) * (1 - time_pressure) * opponent_utility
            else:
                score = self.selfishness * our_utility + (1.0 - self.selfishness) * opponent_utility
        elif self.bid_scoring_metric == "nash_product":
            if self.warmup_counter.is_warmed_up():
                opponent_utility = float(self.opponent_model.get_predicted_utility(bid))
            else:
                opponent_utility = 1.0
            if self.bid_scoring_is_time_dependant:
                time_pressure = self._get_time_pressure()

                # We use log because the score can get really small
                score = self.selfishness * np.log(our_utility + 1e-6) + np.log(opponent_utility + 1e-6) * (1 - self.selfishness) * (1 - time_pressure)
            else:
                # We use log because the score can get really small
                score = self.selfishness * np.log(our_utility + 1e-6) + np.log(opponent_utility + 1e-6) * (1 - self.selfishness)

        else:
            raise ValueError(f"{self.bid_scoring_metric} bid scoring metric is not implemented in score_bid method!")
        # To consider the bids that are better for our own more often
        if our_utility <= opponent_utility:
            if score < 0:
                score *= 2
            else:
                score /= 2

        if self.profile.getReservationBid() is not None and self.profile.getUtility(bid) < self.profile.getUtility(
                self.profile.getReservationBid()) or our_utility <= self.min_utility_to_accept:
            if score < 0:
                score = -10.0
            else:
                score = 0.0

        return score

    def _get_time_pressure(self):
        progress = self.progress.get(time() * 1000)
        return 1.0 - progress ** (1 / self.time_pressure_factor)
