import random

def ca_decision(node, grid, T=4):
    """
    CA-based contention rule with:
      - Small random jitter to avoid sync-ing defers
      - Scaling of penalties by per-BS global_share for cross-tech fairness
      - Original NAV, fairness grant, neighbor-polling, and priority logic
    Returns:
      (penalty_slots, defer_strength)
    """

    # 1) If NAV is active, we must wait AIFS immediately
    if node.base_station.nav_expiry_time > node.env.now:
        return node.aifs_slots, 0

    # 2) If BS has signaled a fairness backoff, defer at least AIFS + CW
    if node.base_station.backoff_event.triggered:
        return node.aifs_slots + node.cw, 0

    # 3) Compute neighbor “busy_score” over last 5 ms
    busy_score = 0.0
    for info in node.neighbor_info.values():
        age = node.env.now - info['last_tx']
        if age <= 0.005:
            busy_score += info['priority'] * info['cw']

    # 4) Compute an “effective T” based on tech & priority
    if node.tech == "WiFi":
        if node.ac == "AC_BE":
            effective_T = T * 1.1   # best‐effort hears busy more
        else:
            effective_T = T * 0.8   # voice hears busy less
    else:  # NR-U
        if node.ac == "NRU_High":
            effective_T = T * 0.9
        else:
            effective_T = T * 1.3

    # 4a) Add a bit of randomness to T so all nodes don't sync
    jitter_factor = random.uniform(0.9, 1.1)  # ±10%
    effective_T *= jitter_factor

    # 5) If channel seems idle enough, go right ahead
    if busy_score <= effective_T:
        node.cw = node.cw_min  # reset CW on success
        return 0, 0

    # 6) Channel is busy → we must back off
    #    Double CW (up to max), then wait at least AIFS
    node.cw = min(node.cw * 2, node.cw_max)
    penalty_slots = node.aifs_slots

    # 6a) Extra penalty for best-effort Wi-Fi
    if node.tech == "WiFi" and node.ac == "AC_BE":
        penalty_slots += int(node.cw * 0.2)

    # 6b) **Scale** the penalty by the BS’s global_share
    #     (if share>1, penalty shrinks; if share<1, it grows)
    penalty_slots = max(
        node.aifs_slots,
        int(penalty_slots / node.base_station.global_share)
    )

    # 6c) **Random jitter** on the backoff slots (0–1 extra slot)
    penalty_slots += random.randint(0, 1)

    # 7) Finally, check if we should *voluntarily* defer for starving neighbors
    defer_strength = 0
    if hasattr(node, 'neighbor_starvation_detected'):
        defer_strength = node.neighbor_starvation_detected()

    return penalty_slots, defer_strength
