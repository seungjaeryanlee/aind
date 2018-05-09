
from sample_players import DataPlayer
import random

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture

        if state.ply_count < 2: self.queue.put(random.choice(state.actions()))
        else:
            # Iterative Deepening
            for i in range(3, 32):
                # self.queue.put(self.alphabeta(state, depth=i))
                self.queue.put(self.negamax_alphabeta(state, depth=i, alpha=float("-inf"), beta=float("inf"), color=1))

    def negamax_root(self, state, depth, color):
        """
        Negamax variant of Minimax
        """
        def negamax(state, depth, color):
            if state.terminal_test(): return color * state.utility(self.player_id)
            if depth <= 0: return color * self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, -negamax(state.result(action), depth - 1, -color))
            return value

        return max(state.actions(), key=lambda x: -negamax(state.result(x), depth - 1, -color))

    def negamax_alphabeta(self, state, depth, alpha, beta, color):
        """
        Negamax variant of Minimax with alpha-beta pruning.
        """
        def negamax(state, depth, alpha, beta, color):
            if state.terminal_test(): return color * state.utility(self.player_id)
            if depth <= 0: return color * self.score(state)
            best_value = float("-inf")
            for action in state.actions():
                value = -negamax(state.result(action), depth - 1, -beta, -alpha, -color)
                best_value = max(best_value, value)
                alpha = max(alpha, value)
                if beta <= alpha: break
            return best_value

        return max(state.actions(), key=lambda x: -negamax(state.result(x), depth - 1, -beta, -alpha, -color))

    def alphabeta(self, state, depth):
        """
        Minimax search with alpha-beta pruning.
        """
        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha: break
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha: break
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, float("-inf"), float("inf")))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
        return len(own_liberties) - len(opp_liberties)
