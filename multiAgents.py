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

#Author: Bryan Tong


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
        evalFunctionScore = successorGameState.getScore()
        for ghost in successorGameState.getGhostPositions():
            distanceToGhost = manhattanDistance(newPos, ghost)
            if (distanceToGhost < 1):
                return -100000
        foodGrid = newFood.asList()
        minFoodDist = float('inf')
        i = 0
        while i < len(foodGrid):
            tempFoodDist = manhattanDistance(newPos, foodGrid[i])
            minFoodDist = min(minFoodDist, tempFoodDist)
            i += 1
        evalFunctionScore = evalFunctionScore + 1/minFoodDist
        return evalFunctionScore

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
        def minimax(depth, agentIndex, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex != 0:
                return findMin(depth, agentIndex, gameState)
            else:
                return findMax(depth, 0, gameState)

        def findMax(depth, agentIndex, gameState):
            maxVal = -float('inf')
            possibleMoves = gameState.getLegalActions(agentIndex)
            i = 0
            while i < len(possibleMoves):
                tempVal = minimax(depth, 1, gameState.generateSuccessor(agentIndex, possibleMoves[i]))
                if tempVal > maxVal:
                    maxVal = tempVal
                i += 1
            return maxVal

        def findMin(depth, agentIndex, gameState):
            nextIndex = agentIndex + 1
            if agentIndex == gameState.getNumAgents()-1:
                nextIndex = 0
                depth += 1
            minVal = float('inf')
            possibleMoves = gameState.getLegalActions(agentIndex)
            i = 0
            while i < len(possibleMoves):
                tempVal = minimax(depth, nextIndex, gameState.generateSuccessor(agentIndex, possibleMoves[i]))
                if tempVal < minVal:
                    minVal = tempVal
                i += 1
            return minVal

        bestMove = Directions.NORTH
        possibleMoves = gameState.getLegalActions()
        max = -float('inf')
        i = 0
        while i < len(possibleMoves):
            temp = minimax(0, 1, gameState.generateSuccessor(0, possibleMoves[i]))
            if temp > max:
                max = temp
                bestMove = possibleMoves[i]
            i += 1

        return bestMove

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def alphabeta(depth, agent, gameState, alpha, beta):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent != 0:
                return findMin(depth, agent, gameState, alpha, beta)
            else:
                return findMax(depth, agent, gameState, alpha, beta)

        def findMin(depth, agentIndex, gameState, alpha, beta):
            minVal = float('inf')
            nextIndex = agentIndex + 1
            if agentIndex == gameState.getNumAgents()-1:
                nextIndex = 0
                depth += 1
            possibleMoves = gameState.getLegalActions(agentIndex)
            i = 0
            while i < len(possibleMoves):
                minVal = min(minVal, alphabeta(depth, nextIndex, gameState.generateSuccessor(agentIndex, possibleMoves[i]), alpha, beta))
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)
                i += 1
            return minVal

        def findMax(depth, agentIndex, gameState, alpha, beta):
            maxVal = float('-inf')
            possibleMoves = gameState.getLegalActions(agentIndex)
            i = 0
            while i < len(possibleMoves):
                maxVal = max(maxVal, alphabeta(depth, 1, gameState.generateSuccessor(agentIndex, possibleMoves[i]), alpha, beta))
                if maxVal > beta:
                    return maxVal
                alpha = max(alpha, maxVal)
                i += 1
            return maxVal

        bestMove = Directions.NORTH
        maxVal = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        possibleMoves = gameState.getLegalActions()
        i = 0
        while i < len(possibleMoves):
            temp = alphabeta(0, 1, gameState.generateSuccessor(0, possibleMoves[i]), alpha, beta)
            if temp > maxVal:
                maxVal = temp
                bestMove = possibleMoves[i]
            if maxVal > beta:
                return maxVal
            alpha = max(alpha, maxVal)
            i += 1
        return bestMove

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

        def expectimax(depth, agentIndex, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex != 0:
                return findAvg(depth, agentIndex, gameState)
            else:
                return findMax(depth, 0, gameState)

        def findMax(depth, agentIndex, gameState):
            maxVal = -float('inf')
            possibleMoves = gameState.getLegalActions(agentIndex)
            i = 0
            while i < len(possibleMoves):
                tempVal = expectimax(depth, 1, gameState.generateSuccessor(agentIndex, possibleMoves[i]))
                if tempVal > maxVal:
                    maxVal = tempVal
                i += 1
            return maxVal

        def findAvg(depth, agentIndex, gameState):
            nextIndex = agentIndex + 1
            if agentIndex == gameState.getNumAgents()-1:
                nextIndex = 0
                depth += 1
            sum = 0
            answer = 0
            possibleMoves = gameState.getLegalActions(agentIndex)
            probability = 1/len(possibleMoves)
            i = 0
            while i < len(possibleMoves):
                tempVal = expectimax(depth, nextIndex, gameState.generateSuccessor(agentIndex, possibleMoves[i]))
                sum += tempVal
                i += 1
            return sum * probability

        bestMove = Directions.NORTH
        possibleMoves = gameState.getLegalActions()
        max = -float('inf')
        i = 0
        while i < len(possibleMoves):
            temp = expectimax(0, 1, gameState.generateSuccessor(0, possibleMoves[i]))
            if temp > max:
                max = temp
                bestMove = possibleMoves[i]
            i += 1

        return bestMove
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I essentially used the same code from my evaluationFunction except that I used the currentGameState
    rather than using the action to get to the successorGameState. First I check to see if the current position
    of Pacman is extremely close to any ghost; if so, I return a very poor score. After that, I find the minimum
    distance to the nearest food pellet, and add the reciprocal of that distance to the currentGameState score
    and return that value as my answer. 
    """
    "*** YOUR CODE HERE ***"

    currentPos = currentGameState.getPacmanPosition()
    evalFunctionScore = currentGameState.getScore()
    for ghost in currentGameState.getGhostPositions():
        distanceToGhost = manhattanDistance(currentPos, ghost)
        if (distanceToGhost < 1):
            return -100000
    foodGrid = currentGameState.getFood().asList()
    minFoodDist = float('inf')
    i = 0
    while i < len(foodGrid):
        tempFoodDist = manhattanDistance(currentPos, foodGrid[i])
        minFoodDist = min(minFoodDist, tempFoodDist)
        i += 1
    evalFunctionScore = evalFunctionScore + 1/minFoodDist
    return evalFunctionScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
