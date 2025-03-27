import json
import logging
from random import randint
from time import time
from typing import cast

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.bidspace.BidsWithUtility import BidsWithUtility
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
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel
from .utils.opponent_models.linear_additive_utility_space_optimizer import LinearAdditiveUtilitySpaceOptimizer, InitializationMode, NormalizationMode
from .utils.warmup_counter import WarmupCounter


class Agent44(DefaultParty):
    """
    Template of a Python geniusweb agent.
    """

    def __init__(self):
        super().__init__()
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
        self.num_samples = 1000  # Number of samples of bid space to take when finding a bid
        self.max_bid: Bid = None

        self.opponent_model_warmup_rounds = 50  # Number of rounds to not accept any offers with utility less than the best bid before conceding other offers
        self.opponent_model_learning_rate = 0.1  # Learning rate for updating the opponent model
        self.opponent_model_epochs = 50
        self.opponent_model_weight_initialization_mode = InitializationMode.CUSTOM  # Initialization mode for the weights of issues in the opponent utility space model
        self.opponent_model_weight_normalization_mode = NormalizationMode.MAX_MIN  # Normalization mode for the weights of issues in the opponent utility space model

        self.opponent_model_value_initialization_mode = InitializationMode.CUSTOM  # Initialization mode for the values of issues in the opponent utility space model
        self.opponent_model_value_normalization_mode = NormalizationMode.CLIP  # Normalization mode for the values of issues in the opponent utility space model

        self.warmup_counter = WarmupCounter(self.opponent_model_warmup_rounds)
        self.selfishness = 0.8  # Selfishness factor for computing wellfare score of a bid based on my utility and opponents estimated utility
        self.time_pressure_factor = 0.1  # Time pressure factor for computing wellfare score of a bid based on my utility and opponents estimated utility
        self.opponent_model: OpponentModel = None

        self.last_offered_bid: Bid = None

    def notifyChange(self, data: Inform):
        # TODO: to be changed
        # Needs to be changed to load the learned stored data or to add some preprocessing when the agent receives Settings
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

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
                                                                      learning_rate=self.opponent_model_learning_rate,
                                                                      epochs=self.opponent_model_epochs,
                                                                      weights_init_mode=self.opponent_model_weight_initialization_mode,
                                                                      weights_norm_mode=self.opponent_model_weight_normalization_mode,
                                                                      values_init_mode=self.opponent_model_value_initialization_mode,
                                                                      values_norm_mode=self.opponent_model_value_normalization_mode,
                                                                      init_utility_space=None if self.opponent_model_weight_initialization_mode != InitializationMode.CUSTOM and self.opponent_model_weight_initialization_mode != InitializationMode.CUSTOM else self.profile)
            self.max_bid = BidsWithUtility.create(self.profile).getExtremeBid(isMax=True)
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
                self.opponent_model.update(self.last_received_bid, self.last_offered_bid)

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
            print(self.opponent_model.get_utility_space_json_dict())
            profile_data = json.dumps(self.opponent_model.get_utility_space_json_dict(), sort_keys=True, indent=4)
            with open(f"{self.storage_dir}/{self.other}_predicted_profile.json", "w") as f:
                f.write(profile_data)

            if isinstance(self.opponent_model, LinearAdditiveUtilitySpaceOptimizer):
                loss_data= json.dumps(self.opponent_model.get_epoch_losses_json_dict(), sort_keys=True, indent=4)
                with open(f"{self.storage_dir}/{self.other}_rounds_loss.json", "w") as f:
                    f.write(loss_data)

            self.logger.log(logging.INFO, "Saved opponents predicted profile: " + self.other)

    ###########################################################################################
    ################################## Example methods below ##################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        # TODO: to be changed
        # Needs to be changed to a smarter condition that determines if the bid is accepted
        if bid is None or not self.warmup_counter.is_warmed_up():
            return False

        # progress of the negotiation session between 0 and 1 (1 is deadline)
        progress = self.progress.get(time() * 1000)

        # very basic approach that accepts if the offer is valued above 0.7 and
        # 95% of the time towards the deadline has passed
        conditions = [
            self.profile.getUtility(bid) > 0.8,
            progress > 0.95,
        ]
        return all(conditions)

    def find_bid(self) -> Bid:
        # TODO: to be changed
        # Needs to be changed to a smarter way to find a bid

        # if not self.warmup_counter.is_warmed_up():
        #     # if the agent is still warming up, find the bid with the highest utility
        #     return self.max_bid

        # compose a list of all possible bids
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        best_bid_score = 0.0
        best_bid = None

        # take self.num_samples attempts to find a bid according to a heuristic score
        for _ in range(self.num_samples):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        return best_bid

    def score_bid(self, bid: Bid, metric: str = "social_wellfare", time_dependant: bool = True) -> float:
        """Calculate heuristic score for a bid.

        Args:
            bid (Bid): Bid to score.
            metric (str, optional): Metric to use for scoring. Defaults to "social wellfare".
            time_dependant (bool, optional): Whether to consider time pressure in scoring. Defaults to True.

        Returns:
            float: Heuristic score of the bid.
        """
        our_utility = float(self.profile.getUtility(bid))

        if self.warmup_counter.is_warmed_up():
            opponent_utility = float(self.opponent_model.get_predicted_utility(bid))
        else:
            opponent_utility = 0.0

        score = 0.0
        if metric == "social_wellfare":
            if time_dependant:
                time_pressure = self._get_time_pressure()
                score = self.selfishness * time_pressure * our_utility + (1.0 - self.selfishness * time_pressure) * opponent_utility
            else:
                score = self.selfishness * our_utility + (1.0 - self.selfishness) * opponent_utility
        elif metric == "nash_product":
            if time_dependant:
                time_pressure = self._get_time_pressure()
                score = our_utility ** (self.selfishness * time_pressure) * opponent_utility ** (1 - self.selfishness * time_pressure)
            else:
                score = (our_utility ** self.selfishness) * (opponent_utility ** (1 - self.selfishness))

        return score

    def _get_time_pressure(self):
        progress = self.progress.get(time() * 1000)
        return 1.0 - progress ** (1 / self.time_pressure_factor)
