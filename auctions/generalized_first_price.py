from auctions.__base__ import Auction
import numpy as np

class GeneralizedFirstPriceAuction(Auction):
    def __init__(self, ctrs, valuations):
        self.ctrs = ctrs
        self.valuations = valuations
        self.n_adv = len(self.ctrs)
        self.n_slots = len(self.valuations)

    def get_winners(self, bids):
        adv_values = self.ctrs*bids
        adv_ranking = np.argsort(adv_values)
        winner = adv_ranking[-self.n_slots:]
        winner = np.flip(winner)
        return winner, adv_values #Reverse winners to get in position 0 the winner, position 1 the second place ...

    def get_payments_per_click(self, winners, values, bids):
        payment = []
        for element in winners:
          if bids[element] > 0:
            payment.append(bids[element])
          else:
            payment.append(0)
        return payment