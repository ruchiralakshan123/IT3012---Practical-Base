# agent.py
import random
from collections import deque
import heapq
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Condition-action agent with no internal memory."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            return 'Up'
        if percept.get('wall_ahead', False):
            return 'Left'
        if percept.get('wall_left', False):
            return 'Right'
        return 'Up'


class ModelBasedAgent:
    """Simple reflex agent with internal memory to avoid repeating the same dead end."""

    def __init__(self):
        self.visited_states = set()
        self.last_action = 'Up'

    def _state_key(self, percept: dict) -> tuple:
        return tuple(sorted(percept.items()))

    def sense_and_act(self, percept: dict) -> str:
        state_key = self._state_key(percept)
        seen_before = state_key in self.visited_states
        self.visited_states.add(state_key)

        if percept.get('food_here', False):
            action = 'Up'
        elif percept.get('wall_ahead', False) and percept.get('wall_left', False):
            action = 'Right'
        elif percept.get('wall_ahead', False) and seen_before:
            action = 'Right' if self.last_action != 'Right' else 'Left'
        elif percept.get('wall_ahead', False):
            action = 'Left'
        elif seen_before:
            action = 'Right' if self.last_action != 'Right' else 'Up'
        else:
            action = 'Up'

        self.last_action = action
        return action


class SearchAgent:
    """Goal-based agent that plans a route to food before moving.

    The environment supplies a static model of the grid in each percept.  The
    agent uses that model to form an offline plan, then returns one action from
    that plan at a time.
    """

    _MOVES = (
        ('Up', (0, 1)),
        ('Right', (1, 0)),
        ('Down', (0, -1)),
        ('Left', (-1, 0)),
    )

    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo.upper()

    @classmethod
    def _neighbours(cls, position, walls, grid_size):
        """Yield legal (action, next_position) pairs in the grid."""
        width, height = grid_size
        for action, (dx, dy) in cls._MOVES:
            next_pos = (position[0] + dx, position[1] + dy)
            if 0 <= next_pos[0] < width and 0 <= next_pos[1] < height and next_pos not in walls:
                yield action, next_pos

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Return a shortest unit-cost path using a FIFO frontier, or None."""
        start, goal = tuple(start_pos), tuple(goal_pos)
        if start == goal:
            return []

        wall_set = {tuple(wall) for wall in walls}
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            position, path = frontier.popleft()
            for action, next_pos in self._neighbours(position, wall_set, grid_size):
                if next_pos in reached:
                    continue
                next_path = path + [action]
                if next_pos == goal:
                    return next_path
                reached.add(next_pos)
                frontier.append((next_pos, next_path))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Return a path using a LIFO frontier, or None if no route exists."""
        start, goal = tuple(start_pos), tuple(goal_pos)
        if start == goal:
            return []

        wall_set = {tuple(wall) for wall in walls}
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            position, path = frontier.pop()
            # Reverse the insertion order because the last item is expanded first.
            for action, next_pos in reversed(list(self._neighbours(position, wall_set, grid_size))):
                if next_pos in reached:
                    continue
                next_path = path + [action]
                if next_pos == goal:
                    return next_path
                reached.add(next_pos)
                frontier.append((next_pos, next_path))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Return a lowest-cost path using a priority queue, or None.

        Every movement costs one in this practical, so UCS and BFS return paths
        with the same number of actions, though their frontiers differ.
        """
        start, goal = tuple(start_pos), tuple(goal_pos)
        if start == goal:
            return []

        wall_set = {tuple(wall) for wall in walls}
        frontier = [(0, start, [])]
        reached = set()

        while frontier:
            cost, position, path = heapq.heappop(frontier)
            if position in reached:
                continue
            reached.add(position)
            if position == goal:
                return path

            for action, next_pos in self._neighbours(position, wall_set, grid_size):
                if next_pos not in reached:
                    heapq.heappush(frontier, (cost + 1, next_pos, path + [action]))

        return None

    def sense_and_act(self, percept: dict) -> str:
        """Make a plan when necessary and return its next action."""
        if not self.plan:
            start = tuple(percept['agent_pos'])
            walls = percept['walls']
            grid_size = percept['grid_size']
            food_positions = [tuple(food) for food in percept['all_food']]

            if not food_positions:
                return 'Stay'

            search_methods = {
                'BFS': self.bfs_search,
                'DFS': self.dfs_search,
                'UCS': self.ucs_search,
            }
            try:
                search = search_methods[self.active_algo]
            except KeyError as error:
                raise ValueError("active_algo must be 'BFS', 'DFS', or 'UCS'.") from error

            # Try nearer pellets first.  If one is enclosed, continue with the
            # next candidate instead of stopping the simulation.
            ordered_food = sorted(food_positions, key=lambda food: abs(food[0] - start[0]) + abs(food[1] - start[1]))
            for goal in ordered_food:
                path = search(start, goal, walls, grid_size)
                if path is not None:
                    self.plan = path
                    break

            if not self.plan:
                return 'Stay'

        return self.plan.pop(0)
