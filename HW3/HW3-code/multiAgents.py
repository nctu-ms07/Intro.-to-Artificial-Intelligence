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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

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
    Your minimax agent
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        score, action = self._minimax(gameState, self.index, 0)
        # End your code (Part 1)
        return action

    def _minimax(self, currentState, currentIndex, currentDepth):
        if currentDepth == self.depth or currentState.isWin() or currentState.isLose():
          return self.evaluationFunction(currentState), ""

        legalActions = currentState.getLegalActions(currentIndex)
        best_score = float('-inf') if currentIndex == 0 else float('inf')
        best_action = ""
        
        for action in legalActions:
          next_state = currentState.getNextState(currentIndex, action)

          if (currentIndex + 1) % next_state.getNumAgents() == self.index:
            score = self._minimax(next_state, self.index, currentDepth + 1)[0]
          else:
            score = self._minimax(next_state, (currentIndex + 1) % next_state.getNumAgents(), currentDepth)[0]

          if currentIndex == 0 and best_score < score:
            best_score = score
            best_action = action

          elif currentIndex != 0 and best_score > score:
            best_score = score
            best_action = action

        return best_score, best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        score, action = self._minimax(gameState, self.index, 0, float('-inf'),float('inf'))
        # End your code (Part 2)
        return action

    def _minimax(self, currentState, currentIndex, currentDepth, alpha, beta):
        if currentDepth == self.depth or currentState.isWin() or currentState.isLose():
          return self.evaluationFunction(currentState), ""

        legalActions = currentState.getLegalActions(currentIndex)
        best_score = float('-inf') if currentIndex == 0 else float('inf')
        best_action = ""
        
        for action in legalActions:
          next_state = currentState.getNextState(currentIndex, action)

          if (currentIndex + 1) % next_state.getNumAgents() == self.index:
            score = self._minimax(next_state, self.index, currentDepth + 1, alpha, beta)[0]
          else:
            score = self._minimax(next_state, (currentIndex + 1) % next_state.getNumAgents(), currentDepth, alpha, beta)[0]

          if currentIndex == 0 and best_score < score:
            best_score = score
            best_action = action
            alpha = max(alpha, best_score)
            if beta < alpha:
              break

          elif currentIndex != 0 and best_score > score:
            best_score = score
            best_action = action
            beta = min(beta, best_score)
            if beta < alpha:
              break

        return best_score, best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        score, action = self._expectimax(gameState, self.index, 0)
        # End your code (Part 3)
        return action

    def _expectimax(self, currentState, currentIndex, currentDepth):
        if currentDepth == self.depth or currentState.isWin() or currentState.isLose():
          return self.evaluationFunction(currentState), ""

        legalActions = currentState.getLegalActions(currentIndex)
        best_score = float('-inf') if currentIndex == 0 else float(0)
        best_action = ""
        
        for action in legalActions:
          next_state = currentState.getNextState(currentIndex, action)

          if (currentIndex + 1) % next_state.getNumAgents() == self.index:
            score = self._expectimax(next_state, self.index, currentDepth + 1)[0]
          else:
            score = self._expectimax(next_state, (currentIndex + 1) % next_state.getNumAgents(), currentDepth)[0]

          if currentIndex == 0 and best_score < score:
            best_score = score
            best_action = action

          elif currentIndex != 0:
            best_score += (score / float(len(legalActions)))

        return best_score, best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function
    """
    # Begin your code (Part 4)
    if currentGameState.isWin():
      return float('inf')
    if currentGameState.isLose():
      return float('-inf')

    pacman = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    manhattanDistanceToClosestFood = min(map(lambda food: util.manhattanDistance(pacman, food), foods))

    scaredGhosts = []
    for ghost in currentGameState.getGhostStates():
      if ghost.scaredTimer > 0:
        scaredGhosts.append(ghost)

    if scaredGhosts:
      manhattanDistanceToClosestScardGhost = min(map(lambda ghost: util.manhattanDistance(pacman, ghost.getPosition()), scaredGhosts))
    else:
      manhattanDistanceToClosestScardGhost = float('inf')
    # End your code (Part 4)
    return 2 * currentGameState.getScore() + 1 / manhattanDistanceToClosestFood - 20 * len(currentGameState.getCapsules()) + 20 / manhattanDistanceToClosestScardGhost - 20 * (len(currentGameState.getLegalActions(0)) == 2)

# Abbreviation
better = betterEvaluationFunction
