def get_moore_neighbor_states(node, grid):
    """
    Gather the activity of up to 8 surrounding neighbors.
    - node.position: (x, y) coordinates of this node in the grid
    - grid: dict mapping (x, y) -> node object
    Returns a list of 8 values (1 if neighbor transmitted recently, else 0).
    """
    states = []
    x, y = node.position
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbor = grid.get((x + dx, y + dy))
            if neighbor:
                states.append(1 if neighbor.just_transmitted() else 0)
            else:
                states.append(0)
    return states


def ca_decision(node, grid, T=4, delta=1):
    """
    Decide the next backoff slots for a node based on its 8 neighbors.

    Args:
        node: the current node
        grid: position->node map
        T: threshold of busy neighbors to allow immediate transmit
        delta: backoff increment when deferring
    Returns:
        int: number of backoff slots (0=>transmit ASAP, or delta)
    """
    neighbor_states = get_moore_neighbor_states(node, grid)
    busy = sum(neighbor_states)
    if busy <= T:
        return 0
    else:
        # fixed increment delta
        return delta