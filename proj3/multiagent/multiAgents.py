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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = 0
        # consider food
        currentfood = currentGameState.getFood().asList()
        fooddist = []
        for food in newFood.asList():
            fooddist.append(manhattanDistance(food, newPos))
        if len(fooddist) > 0:
            score -= min(fooddist) * 2
        if newPos in currentfood:
            score += 50

        # consider ghost
        ghostdist = []
        ghostpos = []
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                ghostdist.append(manhattanDistance(ghost.getPosition(), newPos))
                ghostpos.append(ghost.getPosition())
        if len(ghostdist) > 0:
            score += min(ghostdist)
        if newPos in ghostpos:
            score -= 100
        # print(score)
        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        legalactions = gameState.getLegalActions(0)
        scores = [self.minimaxvalue(gameState.generateSuccessor(0, action), 1, self.depth) for action in legalactions]
        for i in range(len(legalactions)):
            if scores[i] == max(scores):
                return legalactions[i]

    def minimaxvalue(self, gameState: GameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxvalue(gameState, 0, depth)
        else:
            return self.minvalue(gameState, agentIndex, depth)

    def maxvalue(self, gameState: GameState, agentIndex, depth):
        value = -float("inf")
        for aciton in gameState.getLegalActions(0):
            value = max(value, self.minimaxvalue(gameState.generateSuccessor(0, aciton), 1, depth))
        return value

    def minvalue(self, gameState: GameState, agentIndex, depth):
        value = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                value = min(value, self.minimaxvalue(gameState.generateSuccessor(agentIndex, action), 0, depth - 1))
            else:
                value = min(value,
                            self.minimaxvalue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth))
        return value

        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def max_value(self, depth, gameState, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        Max = -1e9
        Max_action = Directions.STOP
        for action in gameState.getLegalActions(0):
            child = gameState.generateSuccessor(0, action)
            value = self.min_value(depth, child, 1, alpha, beta)
            if value > beta:
                Max = value
                Max_action = action
                break
            if value > Max:
                Max = value
                Max_action = action
            alpha = max(alpha, Max)
        if depth == 1:
            return Max_action
        return Max

    def min_value(self, depth, gameState, num, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        Min = 1e9
        for action in gameState.getLegalActions(num):
            child = gameState.generateSuccessor(num, action)
            if num == gameState.getNumAgents() - 1:
                Min = min(Min, self.max_value(depth + 1, child, alpha, beta))
            else:
                Min = min(Min, self.min_value(depth, child, num + 1, alpha, beta))
            if Min < alpha:
                return Min
            beta = min(beta, Min)
        return Min

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(1, gameState, -1e9, 1e9)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def max_value(self, depth, gameState):
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState)
        Max = -1e9
        Max_action = Directions.STOP
        for action in gameState.getLegalActions(0):
            child = gameState.generateSuccessor(0, action)
            value = self.min_value(depth, child, 1)
            if value > Max:
                Max = value
                Max_action = action
        if depth == 1:
            return Max_action
        return Max

    def min_value(self, depth, gameState, num):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        Min = 0
        for action in gameState.getLegalActions(num):
            child = gameState.generateSuccessor(num, action)
            if num == gameState.getNumAgents() - 1:
                Min += self.max_value(depth + 1, child)
            else:
                Min += self.min_value(depth, child, num + 1)
        return Min

    # def expectvalue(self, gameState: GameState, agentIndex):
    #     value = 0
    #     if gameState.isWin() or gameState.isLose():
    #         return self.evaluationFunction(gameState)
    #     if agentIndex == gameState.getNumAgents() - 1:
    #         for action in gameState.getLegalActions(agentIndex):
    #             value += self.evaluationFunction(gameState.generateSuccessor(agentIndex, action))
    #         return value
    #     for action in gameState.getLegalActions(agentIndex):
    #         value += self.expectvalue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1)
    #     return value

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        return self.max_value(1, gameState)

        # expectlist = []
        # for action in gameState.getLegalActions(0):
        #     expectlist.append(self.expectvalue(gameState.generateSuccessor(0, action), 1))
        # for i in range(len(gameState.getLegalActions(0))):
        #     if expectlist[i] == max(expectlist):
        #         return gameState.getLegalActions(0)[i]

        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    # action = Directions.STOP
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = 0
    if currentGameState.isLose():
        return -float('inf')
    #capsules
    num_capsules = len(currentGameState.getCapsules())
    score -= num_capsules * 4

    # consider food
    currentfood = currentGameState.getFood().asList()
    fooddist = []
    for food in currentfood:
        fooddist.append(manhattanDistance(food, currentGameState.getPacmanPosition()))
    if len(fooddist) > 0:
        score -= min(fooddist)
        score -= len(fooddist) * 10

    # consider ghost
    ghostdist = []
    ghostpos = []
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer == 0:
            ghostdist.append(manhattanDistance(ghost.getPosition(), currentGameState.getPacmanPosition()))
            ghostpos.append(ghost.getPosition())
            if len(ghostdist) > 0 and min(ghostdist) < 4:
                score += min(ghostdist) * 2

    # return score + currentGameState.getScore()
    return score
    # util.raiseNotDefined()



# Abbreviation
better = betterEvaluationFunction
