# agent.py
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
    """Breadth-first search agent for the later practical tasks."""

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        from collections import deque

        if start_pos == goal_pos:
            return []

        width, height = grid_size
        wall_set = {tuple(wall) for wall in walls}
        queue = deque([(tuple(start_pos), [])])
        visited = {tuple(start_pos)}
        moves = [
            ('Up', (0, 1)),
            ('Right', (1, 0)),
            ('Down', (0, -1)),
            ('Left', (-1, 0)),
        ]

        while queue:
            position, path = queue.popleft()
            for action, (dx, dy) in moves:
                next_pos = (position[0] + dx, position[1] + dy)
                if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
                    continue
                if next_pos in wall_set or next_pos in visited:
                    continue
                next_path = path + [action]
                if next_pos == tuple(goal_pos):
                    return next_path
                visited.add(next_pos)
                queue.append((next_pos, next_path))

        return None