def ca_decision(node, grid, T=4):
    """
    CA-based contention rule used to decide whether a node should wait or transmit.
    Uses:
      - Neighbor info broadcast
      - Role of the node (Wi-Fi or NR-U, and its priority class)
      - Fairness backoff signals
    Returns:
      (penalty_slots, defer_strength)
      - penalty_slots: how many backoff slots to wait
      - defer_strength: 0 (no defer), 1 (mild), or 2 (full defer due to neighbor starvation)
    """

    # Step 1: If a NAV (Network Allocation Vector) is active, the node must defer immediately
    if node.base_station.nav_expiry_time > node.env.now:
        return node.aifs_slots, 0

    # Step 2: If the BS has issued a fairness backoff grant, defer longer
    if node.base_station.backoff_event.triggered:
        return node.aifs_slots + node.cw, 0

    # Step 3: Check neighbor activity — sum "busy score" from their priority and CW
    busy_score = 0.0
    for info in node.neighbor_info.values():
        age = node.env.now - info['last_tx']
        # Only count transmissions in the last 5ms
        if age <= 0.005:
            busy_score += info['priority'] * info['cw']

    # Step 4: Apply technology and priority-specific tuning to sensitivity threshold (T)
    if node.tech == "WiFi":
        if node.ac == "AC_BE":
            effective_T = T * 1.1   # Best-effort → more sensitive to busy (40% tighter)
        else:  # AC_VO = Voice
            effective_T = T * 0.8   # Voice → slightly more sensitive
    elif node.tech == "NR-U":
        if node.ac == "NRU_High":
            effective_T = T * 0.9   # High-priority NR-U → more tolerant (20% looser)
        else:  # NRU_Low
            effective_T = T * 1.1  # Low-priority NR-U → tighter
    else:
        effective_T = T  # fallback default

    # Step 5: If total busy score is low, we are allowed to transmit immediately
    if busy_score <= effective_T:
        node.cw = node.cw_min  # reset contention window to minimum
        return 0, 0  # no delay, no starvation handling needed

    # Step 6: If channel is too busy, apply backoff penalty and increase CW
    node.cw = min(node.cw * 2, node.cw_max)
    penalty_slots = node.aifs_slots  # wait at least AIFS time
    # Extra penalty for best-effort Wi-Fi users to prioritize others
    if node.tech == "WiFi" and node.ac == "AC_BE":
        penalty_slots += int(node.cw * 0.2)  # add 20% of CW as delay

    # Step 7: Check if the node should voluntarily defer based on starving neighbors
    defer_strength = 0
    if hasattr(node, 'neighbor_starvation_detected'):
        defer_strength = node.neighbor_starvation_detected()

    return penalty_slots, defer_strength

