from abc import ABC, abstractmethod
from decimal import Decimal, getcontext, setcontext

from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain


class OpponentModel(ABC):
    def __init__(self, domain: Domain, **args):
        self.received_bids: list[Bid] = []
        self.offered_bids: list[Bid] = []
        self.domain = domain
        self.args = args

        # Set the decimal precision to 10
        context = getcontext()
        context.prec = 10
        setcontext(context)

    def update(self, received_bid: Bid, offered_bid: Bid = None):
        # keep track of all bids received and sent
        self.received_bids.append(received_bid)
        if offered_bid:
            self.offered_bids.append(offered_bid)
        self._update()

    @abstractmethod
    def get_predicted_utility(self, bid: Bid) -> Decimal:
        pass

    @abstractmethod
    def _update(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(lambda k, v: f'{k}: {v}',self.args.items()))})"