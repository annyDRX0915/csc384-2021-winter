#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files
import math
import os  # for time functions
from search import *  # for search engines
from sokoban import SokobanState, Direction, \
    PROBLEMS  # for Sokoban specific classes and problems


def sokoban_goal_state(state):
    '''
  @return: Whether all boxes are stored.
  '''
    for box in state.boxes:
        if box not in state.storage:
            return False
    return True


def heur_manhattan_distance(state):
    # IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to
    # it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    heuristic = 0
    for b in state.boxes:
        if b in state.storage:
            continue
        dist = float('inf')
        for s in state.storage:
            d = abs(s[0] - b[0]) + abs(s[1] - b[1])
            if d < dist:
                dist = d
        heuristic += dist
    return heuristic


# SOKOBAN HEURISTICS
def trivial_heuristic(state):
    '''trivial admissible sokoban heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state (# of moves required to get) to the goal.'''
    count = 0
    for box in state.boxes:
        if box not in state.storage:
            count += 1
    return count


def heur_alternate(state):
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between
    # the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.\
    storage = list(state.storage)
    boxes = list(state.boxes)
    obstacles = list(state.obstacles)
    '''print("________________________")
    print((state.width, state.height))
    print(state.robots)
    print(storage)
    print(boxes)
    print(obstacles)
    print("basic stuff done")'''
    found = []
    for b in boxes:
        if b in storage:
            found.append(b)
    for b in found:
        boxes.remove(b)
        storage.remove(b)
    # print("new")
    # print(storage)
    # print(boxes)
    # print(found)

    # base case
    # print("all ok")
    if not boxes:
        return 0
    # stuck on corner
    # print("corner")
    if (0, 0) in boxes:
        return float("inf")
    if (0, state.height - 1) in boxes:
        return float("inf")
    if (state.width - 1, 0) in boxes:
        return float("inf")
    if (state.width - 1, state.height - 1) in boxes:
        return float("inf")

    # on side
    # print("side and two together")
    num_left = 0
    num_right = 0
    num_top = 0
    num_down = 0
    for b in boxes:
        if b[0] == 0:
            num_left += 1
            if (0, b[1] + 1) in boxes or (0, b[1] + 1) in obstacles:
                return float("inf")
            if (0, b[1] - 1) in boxes or (0, b[1] - 1) in obstacles:
                return float("inf")
        if b[0] == state.width - 1:
            num_right += 1
            if (state.width - 1, b[1] + 1) in boxes or (
            state.width - 1, b[1] + 1) in obstacles:
                return float("inf")
            if (state.width - 1, b[1] - 1) in boxes or (
            state.width - 1, b[1] - 1) in obstacles:
                return float("inf")
        if b[1] == 0:
            num_top += 1
            if (b[0] + 1, 0) in boxes or (b[0] + 1, 0) in obstacles:
                return float("inf")
            if (b[0] - 1, 0) in boxes or (b[0] - 1, 0) in obstacles:
                return float("inf")
        if b[1] == state.height - 1:
            num_down += 1
            if (b[0] + 1, state.height - 1) in boxes or (
            b[0] + 1, state.height - 1) in obstacles:
                return float("inf")
            if (b[0] - 1, state.height - 1) in boxes or (
            b[0] - 1, state.height - 1) in obstacles:
                return float("inf")
    for b in storage:
        if b[0] == 0:
            num_left -= 1
        if b[0] == state.width - 1:
            num_right -= 1
        if b[1] == 0:
            num_top -= 1
        if b[1] == state.height - 1:
            num_down -= 1
    if num_left > 0 or num_right > 0 or num_top > 0 or num_down > 0:
        return float("inf")

    # print("stuck")
    # check stuck in corner
    for b in boxes:
        if (b[0] - 1, b[1]) in obstacles and (b[0], b[1] - 1) in obstacles:
            return float("inf")
        if (b[0] - 1, b[1]) in obstacles and (b[0], b[1] + 1) in obstacles:
            return float("inf")
        if (b[0] + 1, b[1]) in obstacles and (b[0], b[1] - 1) in obstacles:
            return float("inf")
        if (b[0] + 1, b[1]) in obstacles and (b[0], b[1] + 1) in obstacles:
            return float("inf")

    heur_dist = 0
    # print("not inf")

    for b in boxes:
        dist_s = float('inf')
        stor = (-1, -1)
        for s in storage:
            d = abs(s[0] - b[0]) + abs(s[1] - b[1])
            if d < dist_s:
                dist_s = d
                stor = s
        # print(dist_s)
        sx_s = min(stor[0], b[0])
        sx_l = max(stor[0], b[0])
        sy_s = min(stor[1], b[1])
        sy_l = max(stor[1], b[1])

        dist_r = float('inf')
        rob = (-1, -1)
        for s in state.robots:
            d = abs(s[0] - b[0]) + abs(s[1] - b[1])
            if d < dist_r:
                dist_r = d
                rob = s
        # print(dist_r)
        rx_s = min(rob[0], b[0])
        rx_l = max(rob[0], b[0])
        ry_s = min(rob[1], b[1])
        ry_l = max(rob[1], b[1])

        if abs(rob[0] - stor[0]) < abs(b[0] - stor[0]):
            heur_dist += 5 * (abs(b[0] - stor[0]) - abs(rob[0] - stor[0]))
        if abs(rob[1] - stor[1]) < abs(b[1] - stor[1]):
            heur_dist += 5 * (abs(b[1] - stor[1]) - abs(rob[1] - stor[1]))

        for s in state.obstacles:
            if sx_s <= s[0] <= sx_l and sy_s <= s[1] <= sy_l:
                dist_s += 10
            if rx_s <= s[0] <= rx_l and ry_s <= s[1] <= ry_l:
                dist_r += 10
        heur_dist += dist_s + 10 * dist_r
    # print(heur_dist)
    # print("--------------------------------------------------------------------")
    return heur_dist


def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0


def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.
    Use this function stub to encode the standard form of weighted A* (i.e. g + w*h)

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """

    # Many searches will explore nodes (or states) that are ordered by their f-value.
    # For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue
    # is the hval of the state.
    # You can use this function to create an alternate f-value for states; this must be a function
    # of the state and the weight.
    # The function must return a numeric f-value.
    # The value will determine your state's position on the Frontier list during a 'custom' search.
    # You must initialize your search engine object as a 'custom' search engine if you supply a custom
    # fval function.
    return sN.gval + (weight * sN.hval)


def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of anytime weighted astar algorithm'''
    start_t = os.times()[0]
    end_t = start_t + timebound
    search_eng = SearchEngine(strategy='custom', cc_level='default')
    search_eng.init_search(initial_state, sokoban_goal_state, heur_fn,
                           lambda sN: fval_function(sN, weight))
    timeout = end_t - os.times()[0]
    cost = (float('inf'), float('inf'), float('inf'))
    best = False
    while timeout > 0 and not search_eng.open.empty():
        search_eng.init_search(initial_state, sokoban_goal_state, heur_fn,
                               lambda sN: fval_function(sN, weight))
        path = search_eng.search(timebound, cost)[0]
        if not path:
            return best
        if not best or path.gval < best.gval:
            best = path
            cost = (best.gval, float('inf'), float('inf'))
        weight = weight / 2
        timeout = end_t - os.times()[0]
    return best


def anytime_gbfs(initial_state, heur_fn, timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of anytime greedy best-first search'''
    start_t = os.times()[0]
    end_t = start_t + timebound
    search_eng = SearchEngine('best_first', 'default')
    search_eng.init_search(initial_state, sokoban_goal_state, heur_fn=heur_fn)
    timeout = end_t - os.times()[0]
    cost = (float("inf"), float("inf"), float("inf"))
    best = False
    while timeout > 0 and not search_eng.open.empty():
        path = search_eng.search(timeout, cost)[0]
        if not path:
            return best
        if not best or path.gval < best.gval:
            best = path
            cost = (float("inf"), float("inf"), best.gval)
        timeout = end_t - os.times()[0]
    return best
