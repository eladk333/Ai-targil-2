import ext_plant
import collections
import time
import random
import math

id = ["322587064"]

# Disclaimer I used Google gemini to help with the python syntax and to summarize material of the course.
# I Also used of the help of Ori Sussan and Gilad Carmel to understand stuff about the course and to understand some ideas about this assignment.

class Controller:
    """This class is a controller for the ext_plant game."""

    def __init__(self, game: ext_plant.Game):
        """Initialize controller for given game model."""
        self.game_ref = game
        config = game.get_problem()

        # Extract map configuration
        self.grid_dims = config["Size"]
        self.rows, self.cols = self.grid_dims
        self.obstacles = set(config.get("Walls", []))
        
        # Game parameters
        self.robot_reliability = config["robot_chosen_action_prob"]
        self.completion_bonus = config["goal_reward"]
        self.max_steps = config["horizon"]
        self.p_rewards = config["plants_reward"]
        
        self.bot_capacities = game.get_capacities()
        self.base_plant_needs = config.get("Plants", {}).copy()

        # Pre-calculate average plant values
        self.plant_estimates = {}
        for coord, rewards in self.p_rewards.items():
            avg_val = sum(rewards) / len(rewards) if rewards else 0
            self.plant_estimates[coord] = avg_val

        # Identify malfunctioning robots
        self.faulty_bots = set()
        if "Robots" in config:
            for rid, details in config["Robots"].items():
                if self.robot_reliability.get(rid, 1.0) < 0.01:
                    self.faulty_bots.add((details[0], details[1]))

        # Pathfinding pre-computation
        self.dist_cache = {}
        poi = set(config.get("Plants", {}).keys()) | set(config.get("Taps", {}).keys())
        
        # Create canonical state to find initial bot positions
        self.start_state = self._pack_state(config)
        for bot_info in self.start_state[0]:
            poi.add(bot_info[1]) # Add initial robot positions

        for point in poi:
            self.dist_cache[point] = self._generate_distance_map(point, self.faulty_bots)

        # Strategy flags
        self.global_best_score = -999
        self.cache_table = {}
        self.should_waste_turn = False
        
        # Analyze variance for "burn" strategy decision
        reward_variance = any(
            (max(r) > min(r)) for r in self.p_rewards.values() if r
        )
        
        tot_needs = sum(config["Plants"].values())
        tot_cap = sum(self.bot_capacities.values())
        sim_rounds = self.max_steps / 15
        cap_potential = sim_rounds * tot_cap
        
        is_map_sparse = tot_needs < (cap_potential * 0.5)

        # Heuristic check to see if we should burn a seed
        if reward_variance and not is_map_sparse and self.max_steps < 150 and len(config.get("Robots", {})) < 5:
            base_score = self._simulate_run(config, do_burn=False)
            burn_score = self._simulate_run(config, do_burn=True)
            if burn_score > base_score + 0.5:
                self.should_waste_turn = True

    def _get_idle_move(self, context):
        """Generates a safe move to waste a turn."""
        try:
            data = context.get_problem() if hasattr(context, 'get_problem') else context
            
            # Find the first robot
            rid = min(data["Robots"].keys())
            r_info = data["Robots"][rid]
            curr_r, curr_c = r_info[0], r_info[1]
            h, w = data["Size"]
            walls = set(data.get("Walls", []))

            directions = [("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1)]
            for name, dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in walls:
                    return f"{name} ({rid})"
            return f"WAIT ({rid})"
        except:
            return "LEFT (10)"

    def _simulate_run(self, config, do_burn=False):
        """Runs a lightweight simulation to test strategy viability."""
        try:
            sim = ext_plant.Game(config)
            if do_burn:
                sim.submit_next_action(self._get_idle_move(config))
                sim.submit_next_action("RESET")
            
            count = 0
            while not sim.get_done() and count < config['horizon']:
                st = sim.get_current_state()
                act = self._decide_strategy(st)
                sim.submit_next_action(act)
                count += 1
            return sim.get_current_reward()
        except:
            return -999

    def _pack_state(self, config):
        """Converts raw problem data into a hashable tuple state."""
        b_list = []
        for rid, (r, c, l, _) in config["Robots"].items():
            b_list.append((rid, (r, c), l))
        b_list.sort(key=lambda x: x[0])

        p_list = sorted([(pos, n) for pos, n in config["Plants"].items()], key=lambda x: x[0])
        t_list = sorted([(pos, w) for pos, w in config["Taps"].items()], key=lambda x: x[0])
        
        need_sum = sum(p[1] for p in p_list)
        return (tuple(b_list), tuple(p_list), tuple(t_list), need_sum)

    def _generate_distance_map(self, origin, bad_nodes=None):
        """Performs BFS to map distances from origin to all reachable points."""
        bad_nodes = bad_nodes or set()
        visited = {origin: 0}
        frontier = collections.deque([(origin, 0)])
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while frontier:
            (cr, cc), dist = frontier.popleft()
            for dr, dc in deltas:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    pos = (nr, nc)
                    if pos not in self.obstacles and pos not in bad_nodes and pos not in visited:
                        visited[pos] = dist + 1
                        frontier.append((pos, dist + 1))
        return visited

    def _query_dist(self, src, dest):
        if dest in self.dist_cache and src in self.dist_cache[dest]:
            return self.dist_cache[dest][src]
        return 999

    def _analyze_route(self, bot_info, plant_pos, active_taps, broken_locs, demand):
        """Calculates a score for a specific robot-plant assignment."""
        rid = bot_info.get('rid', 0)
        loc = bot_info['pos']
        cargo = bot_info['load']
        
        # Calculate steps
        if cargo > 0:
            step_cost = self._query_dist(loc, plant_pos)
        else:
            # Find nearest tap first
            nearest_tap_dist = 999
            best_t = None
            for t in active_taps:
                d = self._query_dist(loc, t)
                if d < nearest_tap_dist:
                    nearest_tap_dist = d
                    best_t = t
            
            if best_t:
                step_cost = nearest_tap_dist + 1 + self._query_dist(best_t, plant_pos)
            else:
                step_cost = 999

        p_val = self.plant_estimates[plant_pos]
        
        delivery_amt = min(demand, cargo) if cargo > 0 else min(demand, self.bot_capacities[rid])
        value = delivery_amt * p_val

        # Adjust for probability
        raw_prob = bot_info['prob']
        adj_prob = raw_prob
        
        if self.completion_bonus <= 10:
             if raw_prob > 0.85: adj_prob = 1.0
             else: adj_prob = math.sqrt(raw_prob)
        else:
             if raw_prob >= 0.75: adj_prob = raw_prob * raw_prob
        
        eff_steps = step_cost / max(adj_prob, 0.01)
        score = max(0.1, value - (0.1 * eff_steps))
        final_metric = score * raw_prob
        
        if step_cost >= 900: final_metric = -1.0
        return step_cost, final_metric

    def _find_efficiency_actions(self, state, ticks_remain):
        """Identifies actions that maximize Reward/Time ratio."""
        bots, plants, taps, _ = state
        tap_locs = [p for p, amt in taps if amt > 0]
        
        max_eff = -1.0
        chosen_act = None
        
        for rid, b_pos, b_load in bots:
            cap = self.bot_capacities.get(rid, 0)
            
            for p_pos, needs in plants:
                if needs <= 0 or p_pos not in self.plant_estimates: continue
                
                # Determine current reward tier
                r_list = self.p_rewards.get(p_pos, [])
                if r_list:
                    init_n = self.base_plant_needs.get(p_pos, len(r_list))
                    idx = min(init_n - needs, len(r_list) - 1)
                    curr_val = r_list[idx] if idx < len(r_list) else r_list[-1]
                else:
                    curr_val = 0

                # Case 1: Deliver existing load
                if b_load > 0:
                    d_dist = self._query_dist(b_pos, p_pos)
                    amt = min(needs, b_load)
                    cost = d_dist + amt
                    
                    if cost <= ticks_remain and cost < 900:
                        total_r = amt * curr_val
                        # Check if this finishes the game
                        if (state[3] - amt) <= 0: total_r += self.completion_bonus
                        
                        metric = total_r / (cost + 2.2)
                        if metric > max_eff:
                            max_eff = metric
                            chosen_act = f"POUR ({rid})" if d_dist == 0 else self._step_toward(b_pos, p_pos, rid)

                # Case 2: Refill and deliver
                if b_load < cap and tap_locs:
                    closest_t_dist = 999
                    target_t = None
                    for t in tap_locs:
                        d = self._query_dist(b_pos, t)
                        if d < closest_t_dist:
                            closest_t_dist = d
                            target_t = t
                    
                    if target_t:
                        space = min(cap - b_load, 99)
                        leg2 = self._query_dist(target_t, p_pos)
                        overhead = closest_t_dist + leg2
                        
                        can_fill = (ticks_remain - overhead) // 2
                        if can_fill >= 1:
                            fill_amt = min(space, can_fill)
                            final_load = b_load + fill_amt
                            del_amt = min(needs, final_load)
                            
                            total_cost = closest_t_dist + fill_amt + leg2 + del_amt
                            
                            if total_cost <= ticks_remain and total_cost < 900:
                                # Re-eval reward for this path
                                r_list = self.p_rewards.get(p_pos, [])
                                if r_list:
                                    init_n = self.base_plant_needs.get(p_pos, len(r_list))
                                    idx = min(init_n - needs, len(r_list) - 1)
                                    curr_val = r_list[idx] if idx < len(r_list) else r_list[-1]
                                else:
                                    curr_val = 0
                                    
                                r_fill = del_amt * curr_val
                                if (state[3] - del_amt) <= 0: r_fill += self.completion_bonus
                                
                                metric = r_fill / (total_cost + 2.2)
                                if metric > max_eff:
                                    max_eff = metric
                                    chosen_act = f"LOAD ({rid})" if closest_t_dist == 0 else self._step_toward(b_pos, target_t, rid)
                                     
        return max_eff, chosen_act

    def _step_toward(self, curr, target, rid):
        """Greedy move towards target."""
        moves = { "UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1) }
        for m_name, (dr, dc) in moves.items():
             nxt = (curr[0]+dr, curr[1]+dc)
             if self._query_dist(nxt, target) < self._query_dist(curr, target):
                 return f"{m_name} ({rid})"
        return None

    def _estimate_max_single_route(self, state, time_budget):
        if time_budget <= 0: return -999
        
        bots, plants, taps, _ = state
        tap_list = [p for p, a in taps if a > 0]
        
        broken = set()
        b_data = []
        for rid, pos, l in bots:
            p = self.robot_reliability.get(rid, 1.0)
            if p < 0.01:
                broken.add(pos)
                p = 0.001 
            b_data.append({'rid': rid, 'pos': pos, 'load': l, 'prob': p})

        best_val = -999
        for p_pos, needs in plants:
            if needs <= 0 or p_pos not in self.plant_estimates: continue
            for r in b_data:
                steps, sc = self._analyze_route(r, p_pos, tap_list, broken, needs)
                if steps < time_budget and sc > best_val:
                    best_val = sc
        return best_val

    def _calculate_state_space(self):
        """Approximates the number of reachable states."""
        reachable = set()
        if self.dist_cache:
             for _, v_map in self.dist_cache.items():
                 reachable.update(v_map.keys())
        cells = len(reachable) or (self.rows * self.cols)
        
        complexity = 1.0
        for rid in self.robot_reliability:
             complexity *= (cells * (self.bot_capacities.get(rid, 0) + 1))
        for n in self.base_plant_needs.values():
             complexity *= (n + 1)
        if hasattr(self, 'game_ref'):
             for _, amt in self.game_ref.get_current_state()[2]:
                  complexity *= (amt + 1)
        return complexity

    def _perform_value_iteration(self):
        """Executes VI for small state spaces."""
        q = collections.deque([self.start_state])
        seen = {self.start_state}
        limit = 100000 
        ctr = 0
        
        # BFS to find reachable states
        while q:
            s = q.popleft()
            ctr += 1
            if ctr > limit: return
            
            for act in self._get_valid_moves(s):
                if act == "RESET": continue
                for nxt, _, _ in self._get_outcomes(s, act, force_det=False):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
        
        # VI Update
        self.vi_policy_map = {}
        values = {s: 0.0 for s in seen}
        
        for t in range(1, self.max_steps + 1):
            new_v = values.copy()
            for s in seen:
                if s[3] == 0: 
                    new_v[s] = self.completion_bonus
                    self.vi_policy_map[(t, s)] = "RESET"
                    continue
                    
                best_q = float('-inf')
                best_a = "RESET"
                
                for a in self._get_valid_moves(s):
                    avg_r = 0
                    for nxt, p, r in self._get_outcomes(s, a):
                        avg_r += p * (r + values.get(nxt, 0))
                    
                    if avg_r > best_q:
                        best_q = avg_r
                        best_a = a
                
                new_v[s] = best_q
                self.vi_policy_map[(t, s)] = best_a
            values = new_v

    def choose_next_action(self, state):
        """Public API: Selects the next action."""
        # Burn strategy handling
        if self.should_waste_turn and hasattr(self, 'game_ref'):
             step = self.game_ref.get_current_steps()
             if step == 0: return self._get_idle_move(self.game_ref)
             if step == 1: return "RESET"

        return self._decide_strategy(state)

    def _decide_strategy(self, state):
        """Dispatches to the appropriate solver based on game state."""
        self.cache_table.clear()
        
        now = self.game_ref.get_current_steps()
        remaining = self.max_steps - now

        # Run VI if small enough
        if not hasattr(self, '_vi_done'):
             self._vi_done = True
             if self._calculate_state_space() < 50000:
                  self._perform_value_iteration()

        if hasattr(self, 'vi_policy_map') and (remaining, state) in self.vi_policy_map:
             return self.vi_policy_map[(remaining, state)]
        
        # Dynamic Threshold Calculation
        if not hasattr(self, '_threshold_ready'):
            self._compute_dynamic_thresholds()
            
        # Reset Logic
        reset_score, _ = self._find_efficiency_actions(self.start_state, remaining - 1)
        curr_score, _ = self._find_efficiency_actions(state, remaining)
        
        # Check if we are near end condition
        loaded = sum(r[2] for r in state[0])
        near_finish = loaded >= (sum(self.base_plant_needs.values()) * 0.5)
             
        should_reset = False
        if curr_score == -1: 
             should_reset = True
        elif reset_score > (curr_score + self._reset_tolerance):
             should_reset = True
             
        critical_fail = curr_score < (reset_score * 0.33)
             
        if self._reset_tolerance > 0.1:
             if not critical_fail and (curr_score > reset_score * 0.8):
                 should_reset = False
             can_reset = (remaining > 10)
        else:
             can_reset = (remaining > 5)
        
        if should_reset and can_reset:
             if not near_finish or curr_score == -1:
                  if state != self.start_state:
                       return "RESET"
        
        # Select Search Method
        min_reliability = min(self.robot_reliability.values()) if self.robot_reliability else 1.0

        if min_reliability >= 0.79:
             return self._search_efficient(state)
        return self._search_robust(state)

    def _compute_dynamic_thresholds(self):
        """Calculates internal heuristics for deciding when to reset or farm."""
        potential = sum(need * self.plant_estimates.get(pos, 0) 
                       for pos, need in self.base_plant_needs.items())
        
        denom = self.completion_bonus + potential + 0.1
        
        fleet_cap = max(1, sum(self.bot_capacities.get(r[0], 0) for r in self.start_state[0]))
        avg_dim = (self.rows + self.cols) / 2.0
        tot_need = sum(self.base_plant_needs.values())
        
        est_steps = (tot_need * 2) + ((tot_need / fleet_cap) * 2 * avg_dim)
        mission_eff = (self.completion_bonus + potential) / (max(est_steps, 1) * 1.2)
        
        best_farm_eff = 0
        for pos, need in self.base_plant_needs.items():
            val = self.plant_estimates.get(pos, 0)
            
            # Estimate distance to water
            t_dist = avg_dim
            if pos in self.dist_cache:
                 local_min = 999
                 for tap_tup in self.start_state[2]:
                     t_p = tap_tup[0]
                     if t_p in self.dist_cache[pos]:
                         d = self.dist_cache[pos][t_p]
                         if d < local_min: local_min = d
                 if local_min != 999: t_dist = local_min
            
            # Find best robot for this plant
            best_r_eff = 0
            best_rid = -1
            
            for r_tup in self.start_state[0]:
                 rid = r_tup[0]
                 cap = max(1, self.bot_capacities.get(rid, 1))
                 cycles = max(1, need / cap)
                 prob = self.robot_reliability.get(rid, 1.0)
                 
                 cost_base = 3 + (cycles * 2 * t_dist) + (2 * need)
                 exp_cost = cost_base / prob if prob > 0 else 999999.0
                 
                 adj_eff = (val * need) / exp_cost
                 
                 if adj_eff > best_r_eff:
                     best_r_eff = adj_eff
                     best_rid = rid
                 elif abs(adj_eff - best_r_eff) < 0.001:
                     if prob > self.robot_reliability.get(best_rid, 0):
                          best_rid = rid

            if best_rid == -1:
                 # Fallback
                 best_r_tup = max(self.start_state[0], key=lambda r: self.bot_capacities.get(r[0], 0))
                 best_rid = best_r_tup[0]
            
            cap = max(1, self.bot_capacities.get(best_rid, 1))
            cycles = max(1, need / cap)
            raw_cost = 3 + (cycles * 2 * t_dist) + (2 * need)
            
            raw_eff = (val * need) / raw_cost
            if raw_eff > best_farm_eff:
                best_farm_eff = raw_eff
                self.target_farm_need = need
                self.target_farm_bot = best_rid

        self._reset_tolerance = 0.05 if best_farm_eff > mission_eff else 0.15
        self._threshold_ready = True

    def _search_efficient(self, state):
        """Depth-limited search optimized for high-probability scenarios."""
        curr_step = self.game_ref.get_current_steps()
        left = self.max_steps - curr_step
        
        valid = self._get_valid_moves(state)
        if not valid: return "RESET"

        pruned = self._refine_legal_moves(state, valid, left)
        if not pruned: return "RESET"

        t_start = time.time()
        t_max = 0.5 + (20.0 / self.max_steps)
        
        best_set = []
        self._search_meta = {'start': t_start, 'limit': t_max, 'nodes': 0, 'node_lim': 35000}
        
        try:
            for d in range(1, 20): 
                if (time.time() - t_start) > t_max: break
                
                iter_best = []
                iter_max = float('-inf')
                completed = True
                
                for a in pruned:
                    if (time.time() - t_start) > (t_max * 1.5) and d > 2: 
                        completed = False; break
                         
                    v = self._eval_move_eff(state, a, d, left)
                    if v > iter_max:
                        iter_max = v; iter_best = [a]
                    elif v == iter_max:
                        iter_best.append(a)
                
                if completed and iter_best: best_set = iter_best
                if not completed: break
        except TimeoutError:
             pass
        
        if not best_set: return "RESET"
        pours = [x for x in best_set if x.startswith("POUR")]
        return random.choice(pours) if pours else random.choice(best_set)

    def _eval_move_eff(self, state, action, depth, time_rem):
        meta = self._search_meta
        if (time.time() - meta['start']) > (meta['limit'] * 1.5): raise TimeoutError()

        min_p = min(self.robot_reliability.values()) if self.robot_reliability else 1.0
        det_flag = (min_p >= 0.79) or (depth <= 1)
        
        outcomes = self._get_outcomes(state, action, force_det=det_flag)
        total = 0
        for s_prime, p, r in outcomes:
            total += p * (r + (0.999 * self._expectimax_recurse(s_prime, depth - 1, time_rem - 1)))
        return total

    def _expectimax_recurse(self, state, depth, time_rem):
        meta = self._search_meta
        if (time.time() - meta['start']) > (meta['limit'] * 1.2): raise TimeoutError()
        meta['nodes'] += 1
        if meta['nodes'] > meta['node_lim']: raise TimeoutError()
                
        key = (state, depth, time_rem, 'eff')
        if key in self.cache_table: return self.cache_table[key]

        if state[3] == 0: return self.completion_bonus + 1000
        if time_rem <= 0: return -1000
        if depth == 0: return self._heuristic_balanced(state, time_rem)

        acts = self._get_valid_moves(state)
        if acts: acts = self._refine_legal_moves(state, acts, time_rem)
        if not acts: return self._heuristic_balanced(state, time_rem)
            
        best = float('-inf')
        for a in acts:
            val = self._eval_move_eff(state, a, depth, time_rem)
            if val > best: best = val
        
        self.cache_table[key] = best
        return best

    def _heuristic_balanced(self, state, time_rem):
        """Balanced heuristic function for standard play."""
        bots, plants, taps, needs_total = state
        val_accum = 0
        
        taps_active = [p for p, a in taps if a > 0]
        bot_list = []
        for rid, pos, l in bots:
             p = max(self.robot_reliability.get(rid, 1.0), 0.001)
             bot_list.append({'rid': rid, 'pos': pos, 'load': l, 'prob': p})
             
        dist_penalty = 0
        worst_path = 0 
        
        for p_pos, n in plants:
            if n <= 0 or p_pos not in self.plant_estimates: continue
            
            r_avg = self.plant_estimates[p_pos]
            closest = 999
            
            for b in bot_list:
                d = 999
                if b['load'] > 0:
                    d = self._query_dist(b['pos'], p_pos)
                elif taps_active:
                    tap_legs = [self._query_dist(b['pos'], t) + self._query_dist(t, p_pos) for t in taps_active]
                    if tap_legs: d = min(tap_legs)
                
                if d < closest: closest = d
            
            if closest >= 900: dist_penalty += 1000
            else:
                 dist_penalty += closest
                 if closest > worst_path: worst_path = closest
            
            val_accum += (n * r_avg)

        val_accum -= (needs_total * 50)
        val_accum -= ((dist_penalty + worst_path) * 2.0)
        
        # Farming logic adjustment
        is_farming_mode = hasattr(self, '_reset_tolerance') and self._reset_tolerance <= 0.05
        
        hoard_mult = 15.0 if is_farming_mode else 25.0
        limit_load = getattr(self, 'target_farm_need', 999) if is_farming_mode else 999
        if limit_load == 0: limit_load = max(self.base_plant_needs.values()) if self.base_plant_needs else 999

        for b in bot_list:
            u_load = min(b['load'], time_rem, limit_load)
            
            mult = hoard_mult
            if is_farming_mode and hasattr(self, 'target_farm_bot'):
                 if b['rid'] != self.target_farm_bot: mult = 0.0
            
            val_accum += u_load * mult * b['prob']
            
            if is_farming_mode and b['load'] > limit_load:
                 val_accum -= 5000.0
                 
        return val_accum

    def _search_robust(self, state):
        """Iterative Deepening Search for difficult scenarios."""
        curr_step = self.game_ref.get_current_steps()
        rem = self.max_steps - curr_step
        
        # Safety check against regression
        curr_est = self._estimate_max_single_route(state, rem)
        if curr_est < (self.global_best_score - 0.1) and rem > 10:
             return "RESET"
        
        legal = self._get_valid_moves(state)
        if not legal: return "RESET"

        candidates = self._refine_legal_moves(state, legal, rem)
        if not candidates: return "RESET"

        t0 = time.time()
        limit = 0.5 + (20.0 / self.max_steps)
        
        best_overall = []
        branch_f = max(1, len(candidates))
        budget = 5000
        
        while True:
             if (time.time() - t0) > limit: break
             
             d = 1
             w = branch_f
             while (w * branch_f) < budget and d < 30:
                 w *= branch_f; d += 1
             
             local_max = float('-inf')
             local_set = []
             
             for a in candidates:
                 if (time.time() - t0) > (limit * 1.5): break
                 
                 v = self._eval_robust_step(state, a, d, rem)
                 if v > local_max:
                     local_max = v; local_set = [a]
                 elif v == local_max:
                     local_set.append(a)
             
             if local_set: best_overall = local_set
             if (time.time() - t0) > (limit * 1.5) or d >= 6: break
             budget *= 2
        
        if not best_overall: return "RESET"
        return random.choice(best_overall)

    def _refine_legal_moves(self, state, moves, time_rem):
        """Filters moves to reduce branching factor."""
        # Special farming filter
        if hasattr(self, '_reset_tolerance') and self._reset_tolerance <= 0.05 and hasattr(self, 'target_farm_bot'):
             filt = []
             for m in moves:
                 if m == "RESET":
                      filt.append(m); continue
                 parts = m.split()
                 if len(parts) > 1:
                      rid = int(parts[1].strip("()"))
                      if rid != self.target_farm_bot:
                           if parts[0] == "LOAD": continue
                      filt.append(m)
             if filt: moves = filt

        bots, plants, taps, _ = state
        b_map = {r[0]: (r[1], r[2]) for r in bots}
        
        grouped = {}
        active_p = [p for p, n in plants]
        
        # Precompute nearest plant per robot
        nearest_p = {}
        for rid, (pos, _) in b_map.items():
            best = 999
            for p in active_p:
                 d = self._query_dist(pos, p)
                 if d < best: best = d
            nearest_p[rid] = best

        for m in moves:
            if m == "RESET": continue
            
            seg = m.split()
            atype, rid = seg[0], int(seg[1].strip("()"))
            
            if rid not in b_map: continue
            pos, load = b_map[rid]
            if rid not in grouped: grouped[rid] = []
            
            if atype == "LOAD":
                p_dist = nearest_p.get(rid, 999)
                if load >= time_rem or (load + p_dist) >= time_rem: continue
                grouped[rid].append((m, -1))
                continue

            if atype == "POUR":
                grouped[rid].append((m, -1))
                continue
            
            # Distance heuristic for movement
            targets = active_p if load > 0 else [t for t, a in taps if a > 0]
            if not targets:
                grouped[rid].append((m, 0)); continue
                
            deltas = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}
            dr, dc = deltas[atype]
            nxt = (pos[0]+dr, pos[1]+dc)
            
            min_d = min([self._query_dist(nxt, t) for t in targets]) if targets else 999
            grouped[rid].append((m, min_d))

        # Select best move per robot
        final_list = []
        for rid, m_list in grouped.items():
            m_list.sort(key=lambda x: x[1])
            if m_list:
                top = m_list[0][1]
                for act, score in m_list:
                    if score <= top: final_list.append(act)

        # Sort by urgency
        def ranker(act):
             if act == "RESET": return -999
             try:
                 rid = int(act.split()[1].strip("()"))
                 if rid not in b_map: return 0
                 return b_map[rid][1] + (self.bot_capacities.get(rid, 1) * self.robot_reliability.get(rid, 1.0))
             except: return 0

        final_list.sort(key=ranker, reverse=True)
        return final_list

    def _eval_robust_step(self, state, action, depth, time_rem):
        out = self._get_outcomes(state, action)
        
        # Check deterministic conditions
        is_stable = (self.completion_bonus <= 10) or ((depth <= 1) and (self.completion_bonus <= 20))
        if is_stable:
             for s, p, r in out:
                 if p >= (self.robot_reliability.get(10, 1.0) - 0.01):
                     return r + self._robust_expectimax(s, depth - 1, time_rem - 1)
        
        acc = 0
        for s, p, r in out:
            acc += p * (r + self._robust_expectimax(s, depth - 1, time_rem - 1))
        return acc

    def _robust_expectimax(self, state, depth, time_rem):
        k = (state, depth, time_rem)
        if k in self.cache_table: return self.cache_table[k]

        if state[3] == 0: return self.completion_bonus + 1000
        if time_rem <= 0: return -1000
        if depth == 0: return self._heuristic_cost_based(state, time_rem)

        acts = self._get_valid_moves(state)
        if acts: acts = self._refine_legal_moves(state, acts, time_rem)
        if not acts: return self._heuristic_cost_based(state, time_rem)
            
        best = float('-inf')
        for a in acts:
            v = self._eval_robust_step(state, a, depth, time_rem)
            if v > best: best = v
        
        self.cache_table[k] = best
        return best

    def _heuristic_cost_based(self, state, time_rem):
        """Alternative heuristic focusing on fleet service cost."""
        bots, plants, taps, needs = state
        score = -needs * 100
        
        active_taps = [p for p, a in taps if a > 0]
        fleet = []
        for rid, pos, l in bots:
            p = max(self.robot_reliability.get(rid, 1.0), 0.001)
            fleet.append({'pos': pos, 'load': l, 'prob': p})

        for p_pos, n in plants:
            if n <= 0 or p_pos not in self.plant_estimates: continue
            val = self.plant_estimates[p_pos]
            
            best_w = 99999
            for r in fleet:
                steps = 999
                if r['load'] > 0:
                    steps = self._query_dist(r['pos'], p_pos)
                elif active_taps:
                    legs = [self._query_dist(r['pos'], t) for t in active_taps]
                    if legs:
                         t_dist = min(legs)
                         # Find specific tap for accuracy
                         t_best = active_taps[legs.index(t_dist)]
                         steps = t_dist + self._query_dist(t_best, p_pos)
                
                w_steps = steps / r['prob']
                if w_steps < best_w: best_w = w_steps
            
            if best_w < (time_rem * 10):
                score += (n * val) - (best_w * 0.5)
            else:
                score -= 300

        for r in fleet:
            use = min(r['load'], time_rem)
            score += use * 25.0 * r['prob']
        return score

    def _get_valid_moves(self, state):
        bots, plants, taps, _ = state
        out = []
        locs = {pos for _, pos, _ in bots}
        p_set = {pos for pos, _ in plants}
        t_set = {pos for pos, _ in taps}
        
        vecs = { "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1) }

        for rid, (r, c), load in bots:
            for name, (dr, dc) in vecs.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in self.obstacles and (nr, nc) not in locs:
                        out.append(f"{name} ({rid})")
            
            if (r, c) in t_set and load < self.bot_capacities[rid]:
                out.append(f"LOAD ({rid})")
            
            if (r, c) in p_set and (r, c) in self.p_rewards and load > 0:
                out.append(f"POUR ({rid})")
        
        out.append("RESET")
        return out

    def _get_outcomes(self, state, action, force_det=False):
        if action == "RESET": return [(self.start_state, 1.0, 0)]

        parts = action.split()
        atype, rid = parts[0], int(parts[1].strip("()"))
        
        p_succ = self.robot_reliability.get(rid, 1.0)
        s_succ, r_succ = self._simulate_step(state, rid, atype, ok=True)
        
        if force_det: return [(s_succ, 1.0, r_succ)]
        
        results = [(s_succ, p_succ, r_succ)]
        p_fail = 1.0 - p_succ
        
        if p_fail > 0:
            if atype in ["UP", "DOWN", "LEFT", "RIGHT"]:
                alts = self._get_robot_moves(state, rid)
                others = [m for m in alts if m != atype] + ["STAY"]
                
                sub_p = p_fail / len(others)
                for mv in others:
                    if mv == "STAY":
                        results.append((state, sub_p, 0))
                    else:
                        s_f, _ = self._simulate_step(state, rid, mv, ok=True)
                        results.append((s_f, sub_p, 0))
            elif atype == "POUR":
                s_f, _ = self._simulate_step(state, rid, atype, ok=False)
                results.append((s_f, p_fail, 0))
            elif atype == "LOAD":
                results.append((state, p_fail, 0))
                
        return results

    def _simulate_step(self, state, rid, atype, ok=True):
        b_list = list(state[0])
        p_list = list(state[1])
        t_list = list(state[2])
        cur_need = state[3]
        
        # Locate robot
        idx = next(i for i, x in enumerate(b_list) if x[0] == rid)
        _, (r, c), load = b_list[idx]
        rew = 0
        
        if atype in ["UP", "DOWN", "LEFT", "RIGHT"]:
            deltas = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}
            dr, dc = deltas[atype]
            b_list[idx] = (rid, (r+dr, c+dc), load)
            
        elif atype == "LOAD" and ok:
            t_idx = next(i for i, t in enumerate(t_list) if t[0] == (r,c))
            pos, amt = t_list[t_idx]
            if amt - 1 <= 0: del t_list[t_idx]
            else: t_list[t_idx] = (pos, amt - 1)
            b_list[idx] = (rid, (r,c), load + 1)
            
        elif atype == "POUR":
            if ok:
                p_match = [i for i, p in enumerate(p_list) if p[0] == (r,c)]
                if p_match and (r,c) in self.p_rewards:
                    pidx = p_match[0]
                    ppos, pneed = p_list[pidx]
                    
                    rewards = self.p_rewards[(r,c)]
                    init_need = self.base_plant_needs[(r,c)]
                    consumed = init_need - pneed
                    
                    rew = rewards[consumed] if consumed < len(rewards) else rewards[-1]
                    
                    if pneed - 1 <= 0: del p_list[pidx]
                    else: p_list[pidx] = (ppos, pneed - 1)
                    
                    cur_need -= 1
                    b_list[idx] = (rid, (r,c), load - 1)
                else:
                    b_list[idx] = (rid, (r,c), load - 1)
            else:
                b_list[idx] = (rid, (r,c), load - 1)

        # maintain sort for canonical state
        b_list.sort(key=lambda x: x[0])
        p_list.sort(key=lambda x: x[0])
        t_list.sort(key=lambda x: x[0])
        
        return (tuple(b_list), tuple(p_list), tuple(t_list), cur_need), rew

    def _get_robot_moves(self, state, rid):
        """Helper to find valid directional moves for a specific robot."""
        res = []
        others = {pos for r, pos, _ in state[0] if r != rid}
        
        curr = next(r for r in state[0] if r[0] == rid)
        r, c = curr[1]
        
        dirs = { "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1) }
        for nm, (dr, dc) in dirs.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if (nr, nc) not in self.obstacles and (nr, nc) not in others:
                    res.append(nm)
        return res