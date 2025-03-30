from abc import ABC, abstractmethod
from decimal import Decimal, getcontext, setcontext

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from tudelft_utilities_logging import ReportToLogger


class OpponentModel(ABC):
    def __init__(self, domain: Domain, name: str, logger: ReportToLogger, **args):
        self.received_bids: list[Bid] = []
        self.offered_bids: list[Bid] = []
        self.domain = domain
        self.name = name
        self.logger = logger
        self.args = args

        self.opponent_name = name + "s_opponent"
        # Set the decimal precision to 10
        context = getcontext()
        context.prec = 10
        setcontext(context)

    def set_opponent_name(self, name: str):
        self.opponent_name = name

    def update(self, received_bid: Bid, offered_bid: Bid = None) -> bool:
        # keep track of all bids received and sent
        self.received_bids.append(received_bid)
        if offered_bid:
            self.offered_bids.append(offered_bid)
        return self._update()

    @abstractmethod
    def get_predicted_utility(self, bid: Bid) -> Decimal:
        pass

    @abstractmethod
    def _update(self) -> bool:
        pass

    @abstractmethod
    def get_utility_space(self) -> UtilitySpace:
        pass

    @abstractmethod
    def get_utility_space_json_dict(self) -> dict:
        pass

    def get_repr_json(self) -> dict:
        return {k: self.args[k].__repr__() for k in self.args}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(lambda k: f'{k}: {self.args[k]}', self.args))})"
