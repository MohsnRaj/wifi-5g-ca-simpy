def ca_decision(node, grid, T=4):
    """
    Advanced CA-based contention decision rule with realistic sensing.

    Returns:
      (penalty_slots, defer_strength)
    """
    # If channel under NAV, defer immediately
    if node.base_station.nav_expiry_time > node.env.now:
        return node.aifs_slots, 0  # must return tuple

    # If base station issued fairness backoff grant, defer more
    if node.base_station.backoff_event.triggered:
        return node.aifs_slots + node.cw, 0

    # Compute signal-based busy score from neighbors
    busy_score = 0.0
    x, y = node.position
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor = grid.get((x + dx, y + dy))
            if neighbor:
                pr_dBm = node.base_station.channel.recv_power_dBm(
                    neighbor, node, node.base_station.band
                )
                if pr_dBm >= node.base_station.ed_threshold:
                    busy_score += pr_dBm * neighbor.priority_weight

     # Channel idle → no penalty (but use a lower T for Wi-Fi BE)
     # so BE sees “busy” more often than VO or NR-U.
    effective_T = T
    if node.tech=="WiFi" and node.ac=="AC_BE":
        effective_T = T * 0.8   # 20% tighter threshold
    if node.tech == "NR-U" and node.ac == "NRU_Low":
        effective_T = T * 0.85  # secondary NRU gets tighter threshold
    if busy_score <= effective_T:
        node.cw = node.cw_min
        return 0, 0

    # Channel busy → exponential backoff and AIFS penalty
    node.cw = min(node.cw * 2, node.cw_max)
    # extra backoff penalty in #slots for BE so they wait longer
    penalty_slots = node.aifs_slots
    if node.tech=="WiFi" and node.ac=="AC_BE":
        penalty_slots += int(node.cw * 0.2)  # add 20% of CW as extra defer

    defer_strength = 0
    if hasattr(node, 'neighbor_starvation_detected'):
        defer_strength = node.neighbor_starvation_detected()

    return penalty_slots, defer_strength
