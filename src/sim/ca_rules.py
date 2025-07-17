# Implements an 8-neighbor (Moore) sampler and CA decision rule for Wi-Fi/5G coexistence

def get_moore_neighbor_states(node, grid):
    """
    Gather the activity of up to 8 surrounding neighbors.
    - node.position: (x, y) coordinates of this node in the grid
    - grid: dict mapping (x, y) -> node object
    Returns a list of 8 values (1 if neighbor transmitted recently, else 0).
    """
    states = []
    x, y = node.position
    # Loop over relative positions in a 3x3 block
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # Skip the node itself
            if dx == 0 and dy == 0:
                continue
            neighbor = grid.get((x + dx, y + dy))
            if neighbor:
                # Assume neighbor.just_transmitted() tells if it sent last slot
                states.append(1 if neighbor.just_transmitted() else 0)
            else:
                # No neighbor at this position => treat as idle (0)
                states.append(0)
    return states


def ca_decision(node, grid, T=4, delta=1):
    """
    Decide the next backoff timer for a node based on its 8 neighbors.

    Args:
        node: the current node, with attributes:
            - .position: (x, y) tuple
            - .backoff: current backoff counter (int)
        grid: dict mapping (x, y) -> node, used by the sampler
        T: threshold of busy neighbors to allow immediate transmit
        delta: amount to add to backoff when deferring

    Returns:
        int: the new backoff value (0 to transmit ASAP, or increased)
    """
    # Get the list of neighbor activity states
    neighbor_states = get_moore_neighbor_states(node, grid)
    # Count how many neighbors were busy
    busy = sum(neighbor_states)
    # If few neighbors busy (<= T), transmit immediately
    if busy <= T:
        return 0  # zero backoff => send in next opportunity
    # Otherwise, defer and increase backoff
    return node.backoff + delta

# Example usage:
#   neighbor_states = get_moore_neighbor_states(my_node, node_grid)\#
#   next_backoff = ca_decision(my_node, node_grid, T=4, delta=2)
