# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        for food in newFood.asList():
            md = util.manhattanDistance(newPos, food)
            if md > 0:
                score += 1.0 / md

        for i in range(len(newGhostStates)):
            md = util.manhattanDistance(newPos, newGhostStates[i].getPosition())
            if newScaredTimes[i] > 5 and md > 0:
                score += 1.0 / md
            elif md > 0:
                score -= 1.0 / md

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def DFMinMax(gameState, agent, depth):
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agent)
            value = float('-inf') if agent == 0 else float('inf')
            max_action = None
            next_agent = (agent + 1) % gameState.getNumAgents()
            for action in actions:
                next_state = gameState.generateSuccessor(agent, action)
                if agent == 0:
                    next_value = DFMinMax(next_state, next_agent, depth)
                    if next_value > value:
                        value = next_value
                        max_action = action
                else:
                    next_value = DFMinMax(next_state, next_agent, depth if next_agent != 0 else depth + 1)
                    value = min(value, next_value)
            if depth == 0 and agent == 0:
                return max_action
            else:
                return value

        return DFMinMax(gameState, 0, 0)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AlphaBeta(gameState, alpha, beta, agent, depth):
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agent)
            value = float("-inf") if agent == 0 else float("inf")
            max_action = None
            next_agent = (agent + 1) % gameState.getNumAgents()
            for action in actions:
                next_state = gameState.generateSuccessor(agent, action)
                if agent == 0:
                    next_value = AlphaBeta(next_state, alpha, beta, next_agent, depth)
                    if next_value > value:
                        value = next_value
                        max_action = action
                    if value >= beta:
                        break
                    alpha = max(alpha, value)
                else:
                    next_value = AlphaBeta(next_state, alpha, beta, next_agent, depth if next_agent != 0 else depth + 1)
                    value = min(value, next_value)
                    if value <= alpha:
                        break
                    beta = min(beta, value)
            if depth == 0 and agent == 0:
                return max_action
            else:
                return value

        return AlphaBeta(gameState, float('-inf'), float('inf'), 0, 0)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def Expectimax(gameState, agent, depth):
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agent)
            value = float('-inf') if agent == 0 else 0
            max_action = None
            next_agent = (agent + 1) % gameState.getNumAgents()
            for action in actions:
                next_state = gameState.generateSuccessor(agent, action)
                if agent == 0:
                    next_value = Expectimax(next_state, next_agent, depth)
                    if next_value > value:
                        value = next_value
                        max_action = action
                else:
                    next_value = Expectimax(next_state, next_agent, depth if next_agent != 0 else depth + 1)
                    value += next_value
            if depth == 0 and agent == 0:
                return max_action
            elif agent == 0:
                return value
            else:
                return float(value) / float(len(actions))

        return Expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    curr_pos = currentGameState.getPacmanPosition()
    curr_food = currentGameState.getFood()
    curr_wall = currentGameState.getWalls()
    curr_ghost_state = currentGameState.getGhostStates()
    curr_ghost_state_times = [ghostState.scaredTimer for ghostState in curr_ghost_state]


    score = currentGameState.getScore()

    for food in curr_food.asList():
        md = util.manhattanDistance(curr_pos, food)
        if md > 0:
            score += 2.0 / md

    for i in range(len(curr_ghost_state)):
        md = util.manhattanDistance(curr_pos, curr_ghost_state[i].getPosition())
        if curr_ghost_state_times[i] > 5 and md > 0:
            score += (2.0 / md) * curr_ghost_state_times[i]
        elif md > 0:
            score -= 1.0 / md

    return score

# Abbreviation
better = betterEvaluationFunction
