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
        oldfood = currentGameState.getFood().asList()
        min = util.manhattanDistance(oldfood[0],newPos)
        point = oldfood[0]
        for i in oldfood:
            if(util.manhattanDistance(i,newPos) < min):
                min = util.manhattanDistance(i,newPos)
                point =i
        foodscore = min

        ghosts = [ghost.getPosition() for ghost in newGhostStates]
        mini = util.manhattanDistance(ghosts[0], newPos)
        pos = ghosts[0]
        for i in ghosts:
            if(util.manhattanDistance(i,newPos) < mini):
                mini = util.manhattanDistance(i,newPos)
                pos = i

        if(util.manhattanDistance(newPos , pos) == 0):
            return -109
        else:
            Ghostscore = 2.7*-1.0/mini

        capsules = len(successorGameState.getCapsules())
        foodsleft = len(newFood.asList())
        if foodscore ==0:
            totalscore = Ghostscore - 4*capsules -2*foodsleft
        else:
            totalscore = Ghostscore + 1/foodscore - 4*capsules - 2*foodsleft

        "*** YOUR CODE HERE ***"
        return totalscore
        #return successorGameState.getScore()

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
        s = gameState
        d = self.depth
        legalactions = s.getLegalActions()
        if len(legalactions) == 0:
            return self.evaluationFunction(s)

        sucstate = []
        values = []
        for i in legalactions:
            sucstate.append(s.generateSuccessor(0,i))
            values.append(self.f(s.generateSuccessor(0,i), 1, d))
        ind = values.index(max(values))
        return legalactions[ind]
        #util.raiseNotDefined()

    def f(self, state, agent, d):
        legalactions = state.getLegalActions(agent)
        numagents = state.getNumAgents()
        V = []
        if state.isWin() or state.isLose() or len(legalactions) == 0:
            return self.evaluationFunction(state)

        for action in legalactions:
            ssct = state.generateSuccessor(agent, action)
            if agent == numagents-1 :
                if d == 1:
                    V.append(self.evaluationFunction(ssct))
                    continue
                V.append(self.f(ssct, 0, d-1))
            else:
                V.append(self.f(ssct, agent+1, d))

        if agent == 0:
            return max(V)
        return min(V)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        legalactions = gameState.getLegalActions()
        beta = float('inf')
        acts= [float('-inf'),legalactions[0]]
        for act in legalactions:
            v = self.minValue(gameState.generateSuccessor(0,act), 0, alpha, beta)
            if (acts[0] < v):
                acts[0] =v
                acts[1]= act
            if acts[0] > beta:
                return act
            alpha = max(alpha, acts[0])
        return acts[1]

    def maxValue(self, state, depth , alpha, beta):
        legalactions = state.getLegalActions()
        maximum = float('-inf')
        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluationFunction(state)
    
        for action in legalactions:
            v = self.minValue(state.generateSuccessor(0,action),depth,alpha,beta)
            maximum = max(maximum,v)
            if maximum > beta:
                return maximum
            alpha = max(alpha, maximum)
        return maximum

    def minValue(self, state, depth, alpha, beta, agent=1):
        numagents = state.getNumAgents()
        minimum = float('inf')
        legalactions = state.getLegalActions(agent)

        if state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        for action in legalactions:
            next = state.generateSuccessor(agent,action)
            if agent == numagents-1:
                v = self.maxValue(next, depth+1, alpha, beta)
                minimum = min(minimum,v)
            else:
                v = self.minValue(next,depth,alpha,beta,agent+1)
                minimum = min(minimum,v)
            if minimum < alpha:
                return minimum
            beta = min(beta, minimum)
        return minimum

        

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
        legalactions = gameState.getLegalActions()
        acts= [float('-inf'),legalactions[0]]
        for act in legalactions:
            v = self.chanceValue(gameState.generateSuccessor(0,act), 0)
            if (acts[0] < v):
                acts[0] =v
                acts[1]= act
        return acts[1]
        util.raiseNotDefined()
    
    def maxValue(self, state, depth):
        legalactions = state.getLegalActions()
        maximum = float('-inf')
        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluationFunction(state)
    
        for action in legalactions:
            v = self.chanceValue(state.generateSuccessor(0,action),depth)
            maximum = max(maximum,v)
        return maximum
    
    def chanceValue(self, state, depth, agent=1):
        numagents = state.getNumAgents()
        legalactions = state.getLegalActions(agent)
        chance =0
        if state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        probab = 1/len(legalactions)
        for action in legalactions:
            next = state.generateSuccessor(agent,action)
            if agent == numagents-1:
                v= probab*self.maxValue(next,depth+1)
                chance = chance + v
            else:
                v = probab*self.chanceValue(next,depth,agent+1)
                chance = chance + v
        return chance


    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
