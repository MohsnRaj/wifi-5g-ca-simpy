import numpy as np
# def get_moore_neighbor_states(node, grid):
#     """
#     Gather the activity of up to 8 surrounding neighbors.
#     - node.position: (x, y) coordinates of this node in the grid
#     - grid: dict mapping (x, y) -> node object
#     Returns a list of 8 values (1 if neighbor transmitted recently, else 0).
#     """
#     states = []
#     x, y = node.position
#     for dx in [-1, 0, 1]:
#         for dy in [-1, 0, 1]:
#             if dx == 0 and dy == 0:
#                 continue
#             neighbor = grid.get((x + dx, y + dy))
#             if neighbor:
#                 states.append(1 if neighbor.just_transmitted() else 0)
#             else:
#                 states.append(0)
#     return states

    

def ca_decision(node, grid, T=4, delta=1):
    # recompute slot_time exactly as in run()
    base_slot = 25e-6 if node.tech=='NR-U' else 9e-6
    slot_time = base_slot / node.base_station.global_share

    window = slot_time  #—or maybe delta * slot_time if you prefer

    states = []
    x, y = node.position

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx==0 and dy==0: continue
            neighbor = grid.get((x+dx, y+dy))
            if neighbor:
                busy = 1 if neighbor.just_transmitted(window) else 0
                states.append(busy * neighbor.priority_weight)
            else:
                states.append(0)

    busy_score = sum(states)    
    # print(f"[CA] {node.name}: busy_score={busy_score}  USING T={T}", flush=True)
    if busy_score <= T:
        return 0
    else:
        # base_penalty = 1.0 / max(0.5, node.priority_weight)  
        adjusted_delta = int(np.ceil(delta * (2.5 - min(2.0, node.priority_weight)))) 
        # print(f"[CA] {node.name}: busy_score={busy_score} → adaptive delta={adjusted_delta}", flush=True)
        return adjusted_delta
