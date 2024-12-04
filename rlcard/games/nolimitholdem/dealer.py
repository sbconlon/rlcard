from rlcard.games.base import Card
from rlcard.games.limitholdem import Dealer


class NolimitholdemDealer(Dealer):
    # Removes a card from the deck
    def remove_card(self, card):
        assert(isinstance(card, Card))
        assert(card in self.deck)
        self.deck.remove(card)
