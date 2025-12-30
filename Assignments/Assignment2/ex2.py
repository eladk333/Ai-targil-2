import ext_plant
import collections
import random

id = ["322587064"]

class Controller:
    """This class is a controller for the ext_plant game."""

    def __init__(self, game: ext_plant.Game):
        """Initialize controller for given game model."""
        self.original_game = game
        
        problem = game.get_problem() # Dictionary for the map, rules and objects

        self.rows, self.cols = problem["Size"] # Grid dim

        self.walls = set(problem.get("Walls", [])) # Convert list to set for optimizite of code

        self.robot_probs = problem["robot_chosen_action_prob"] # Dictionary for robot proabiblity 

        self.goal_reward = problem["goal_reward"]
        self.horizon = problem["horizon"]
        self.plants_reward = problem["plants_reward"]

        self.max_capacity = game.get_capacities()

        # Calc avg rewards
        self.plant_values = {}
        for pos, rewards in self.plants_reward.items():
            if rewards:                
                self.plant_values[pos] = sum(rewards) / len(rewards)
            else:
                self.plant_values[pos] = 0

        # If robot has low prob of moving we treat it like a wall
        self.broken_robots = set()
        if "Robots" in problem:
            for rid, data in problem["Robots"].items():
                prob = self.robot_probs.get(rid, 1.0)                                
                if prob < 0.01:                    
                    self.broken_robots.add((data[0], data[1]))
        
        self.initial_state = self._build_initial_state(problem)

        # Calc bfs dist
        self.dist_cache = {}
        self.key_locations = set(problem.get("Plants", {})) | set(problem.get("Taps", {}))
        if "Robots" in problem:
            for r_vals in problem["Robots"].values():                
                self.key_locations.add((r_vals[0], r_vals[1]))

        for loc in self.key_locations:
            self.dist_cache[loc] = self.run_bfs_flood(loc)

        # Our benchmark for reset
        self.initial_best_route_val = self.get_best_route_val(self.initial_state, self.horizon)

        # We save the states so we won't have to calculate them twice
        self.memo = {}

    def _build_initial_state(self, problem):
        """ Reconstructs the canonical initial state tuple. """
        robots = []
        for rid, (r, c, load, _) in problem["Robots"].items():
            robots.append((rid, (r, c), load))
        robots.sort(key=lambda x: x[0])
        
        plants = []
        for pos, need in problem["Plants"].items():
            plants.append((pos, need))
        plants.sort(key=lambda x: x[0])
        
        taps = []
        for pos, water in problem["Taps"].items():
            taps.append((pos, water))
        taps.sort(key=lambda x: x[0])
        
        total_need = sum(p[1] for p in plants)
        return (tuple(robots), tuple(plants), tuple(taps), total_need)
    
    def get_action_value(self, state, action, depth, time_left):
        """ Calculates the Expected Value (Q-Value) of a specific action. """
        outcomes = self.get_transitions(state, action)
        expected = 0
        for next_s, prob, rew in outcomes:
            # Recursive call to expectimax for the next state
            expected += prob * (rew + self.expectimax(next_s, depth - 1, time_left - 1))
        return expected
    
    def is_line_blocked(self, start, end, broken_positions):
        """ 
        Checks if a broken robot is strictly between start and end in a straight line. 
        Crucial for the heuristic to avoid phantom paths.
        """
        r1, c1 = start
        r2, c2 = end
        
        # Optimization: Only check if aligned (same row or col)
        if r1 == r2: # Horizontal
            min_c, max_c = min(c1, c2), max(c1, c2)
            for br, bc in broken_positions:
                if br == r1 and min_c < bc < max_c: return True
        elif c1 == c2: # Vertical
            min_r, max_r = min(r1, r2), max(r1, r2)
            for br, bc in broken_positions:
                if bc == c1 and min_r < br < max_r: return True
        return False

    def calculate_route_metrics(self, r_data, p_pos, active_taps, current_broken):
        """ Helper: Calculates steps and score for a specific robot -> plant route. """
        raw_steps = 999
        
        if self.is_line_blocked(r_data['pos'], p_pos, current_broken):
            raw_steps = 999
        else:
            if r_data['load'] > 0:
                # Robot has water, go straight to plant
                raw_steps = self.get_distance(r_data['pos'], p_pos)
            elif active_taps:
                # Robot needs water, go to tap then plant
                min_tap = 999
                best_tap = None
                for t in active_taps:
                    if self.is_line_blocked(r_data['pos'], t, current_broken): continue
                    d = self.get_distance(r_data['pos'], t)
                    if d < min_tap:
                        min_tap = d
                        best_tap = t
                if best_tap and not self.is_line_blocked(best_tap, p_pos, current_broken):
                    raw_steps = min_tap + 1 + self.get_distance(best_tap, p_pos)

        avg_reward = self.plant_values[p_pos]
        
        # Base Score Formula
        base_score = avg_reward - (0.1 * raw_steps)
        if base_score < 1: base_score = 1 
        
        final_score = base_score * r_data['prob']
        if raw_steps >= 900: final_score = -1.0
            
        return raw_steps, final_score

    def get_best_route_val(self, state, time_left):
        """ Calculates the score of the SINGLE BEST route available. """
        if time_left <= 0: return -999
        
        robots_t, plants_t, taps_t, _ = state
        active_taps = [pos for pos, amt in taps_t if amt > 0]
        
        # Identify broken robots for blockage
        current_broken = set()
        robot_data = []
        for rid, pos, load in robots_t:
            prob = self.robot_probs.get(rid, 1.0)
            if prob < 0.01:
                current_broken.add(pos)
                prob = 0.001 
            robot_data.append({'pos': pos, 'load': load, 'prob': prob})

        max_route_val = -999
        
        for p_pos, need in plants_t:
            if need <= 0 or p_pos not in self.plant_values: continue
            
            for r in robot_data:
                steps, score = self.calculate_route_metrics(r, p_pos, active_taps, current_broken)
                if steps < time_left:
                    if score > max_route_val:
                        max_route_val = score
                    
        return max_route_val

    # Calc bfs from the origion to evvery reachable cell
    def run_bfs_flood(self, origin):
        distance_map = {origin: 0}
        frontier = collections.deque([(origin, 0)])
        grid_h, grid_w = self.rows, self.cols
        obstacles = self.walls
        
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        
        # Runs until there are no more cells we can go to
        while frontier:
            (r, c), dist = frontier.popleft()
            next_dist = dist + 1
            # Runs over each direction of the robbot
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc                
                # Check bounds
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    neighbor = (nr, nc)
                    # Checks the cell is not a wall and we didn'nt visit it                                        
                    if neighbor not in obstacles and neighbor not in distance_map:
                        distance_map[neighbor] = next_dist
                        frontier.append((neighbor, next_dist))
        
        return distance_map
    
    def get_distance(self, origin, destination):
        distances = self.dist_cache.get(destination)        
        if distances is not None:            
            return distances.get(origin, 999)
            
        return 999
    
    # Gets a state and generates all the legal actions we can take
    def get_legal_actions(self, state):
        current_robots, current_plants, current_taps, _ = state

        actions = []

        grid_h, grid_w = self.rows, self.cols
        static_walls = self.walls
        capacity_map = self.max_capacity

        occupied_cells = {r[1] for r in current_robots}
        plant_locations = {p[0] for p in current_plants}
        tap_locations = {t[0] for t in current_taps}

        move_deltas = (
            ("UP", -1, 0), ("DOWN", 1, 0), 
            ("LEFT", 0, -1), ("RIGHT", 0, 1)
        )

        # Loops over all robots on the board
        for rid, (r, c), load in current_robots:            
            # Loops over the directions of the robot
            for name, dr, dc in move_deltas:
                nr, nc = r + dr, c + dc                
                # Check Bounds
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    target = (nr, nc)
                    
                    # Check cell is empty so the move is  legal
                    if target not in static_walls and target not in occupied_cells:
                        actions.append(f"{name} ({rid})")
            
            current_pos = (r, c)
            
            # If it is legal to load we add it to the actions list.
            if current_pos in tap_locations and load < capacity_map[rid]:
                actions.append(f"LOAD ({rid})")
            
            # If it is legal to load we add it to the actions list.
            if current_pos in plant_locations and load > 0:
                if current_pos in self.plants_reward: # Sanity check
                    actions.append(f"POUR ({rid})")

        # we always allow to reset
        actions.append("RESET")
        
        return actions
    
    def apply_deterministic(self, state, rid, action, success=True):
        robots, plants, taps, total_need = state

        r_idx = -1
        curr_r_data = None

        # Loops over the robot to find the one with the correct id
        for i, r in enumerate(robots):
            if r[0] == rid:
                r_idx = i # Store the index of the robot we were given the id for
                curr_r_data = r # Store the full data tuple of the robot
                break
        
        _, (r, c), load = curr_r_data # Unpack the robots data
        reward = 0.0

        # Helper functon to update the robot's tuple without list conversion and also keeps the tuple sorted by id
        def update_robot(new_pos, new_load):
            return robots[:r_idx] + ((rid, new_pos, new_load),) + robots[r_idx+1:]
        
        # Move actions
        if action in ("UP", "DOWN", "LEFT", "RIGHT"):            
            deltas = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
            dr, dc = deltas[action]
            robots = update_robot((r + dr, c + dc), load)

        # Load actions
        elif action == "LOAD" and success:
            # Find the tap at the robot's location
            for i, (t_pos, t_amt) in enumerate(taps):
                if t_pos == (r, c):                    
                    if t_amt > 1:
                        taps = taps[:i] + ((t_pos, t_amt - 1),) + taps[i+1:]
                    else:
                        taps = taps[:i] + taps[i+1:]
                                        
                    robots = update_robot((r, c), load + 1)
                    break

        # Pour actions
        elif action == "POUR":
            if success:
                # Find the plant at the robot's location
                for i, (p_pos, p_need) in enumerate(plants):
                    if p_pos == (r, c):
                        reward = self.plant_values.get(p_pos, 0)
                        
                        if p_need > 1:
                            plants = plants[:i] + ((p_pos, p_need - 1),) + plants[i+1:]
                        else:
                            plants = plants[:i] + plants[i+1:]
                        
                        total_need -= 1
                        robots = update_robot((r, c), load - 1)
                        break
            else:
                robots = update_robot((r, c), load - 1)

        return (robots, plants, taps, total_need), reward

  
    def choose_next_action(self, state):
        """ Choose the next action given a state."""
        raise ValueError("Fill the function")
    
    # Returns a list of all possible outcomes
    def get_transitions(self, state, action):

        if action == "RESET":
            return [(self.initial_state, 1.0, 0)]

        parts = action.split()
        cmd = parts[0]
 
        rid = int(parts[1][1:-1])

        p_success = self.robot_probs.get(rid, 1.0)
        p_fail = 1.0 - p_success
        
        transitions = []

        # Adds the version where the actions successd
        s_prime, reward = self.apply_deterministic(state, rid, cmd, success=True)
        transitions.append((s_prime, p_success, reward))

        # Adds the version where the actions failed
        if p_fail > 0:            
            # Move actions
            if cmd in ("UP", "DOWN", "LEFT", "RIGHT"):
                robots_t = state[0]
                occupied = {r[1] for r in robots_t if r[0] != rid}
                
                curr_r = next(r for r in robots_t if r[0] == rid)
                r, c = curr_r[1]
                
                failures = []
                deltas = (("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1))
                
                h, w = self.rows, self.cols
                
                for move_name, dr, dc in deltas:
                    if move_name == cmd:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    # Check bounds, walls and collisions
                    if 0 <= nr < h and 0 <= nc < w:
                        target = (nr, nc)
                        if target not in self.walls and target not in occupied:
                            failures.append(move_name)
                
                # Stay is always an option in failure
                failures.append("STAY")
                
                # Distribute the failure mass evenly among alternatives
                p_split = p_fail / len(failures)
                
                for alt_move in failures:
                    if alt_move == "STAY":
                        transitions.append((state, p_split, 0))
                    else:
                        # Recalculate deterministic outcome for the slip
                        s_slip, _ = self.apply_deterministic(state, rid, alt_move, success=True)
                        transitions.append((s_slip, p_split, 0))

            # Pouring action
            elif cmd == "POUR":
                s_fail, _ = self.apply_deterministic(state, rid, cmd, success=False)
                transitions.append((s_fail, p_fail, 0))

            # Loadign action
            elif cmd == "LOAD":
                transitions.append((state, p_fail, 0))
                
        return transitions
    
    def heuristic(self, state, time_left):
        """
        Estimates the strategic value of a state.
        Includes blockage checks to avoid optimistic estimates.
        """
        robots, plants, taps, total_need = state
        score = -total_need * 100.0
        
        tap_positions = [t[0] for t in taps if t[1] > 0]
        fleet = []
        broken_locs = set()
        
        for rid, pos, load in robots:
            p = self.robot_probs.get(rid, 1.0)
            if p < 0.01:
                broken_locs.add(pos)
                p = 0.001
            
            usable = load if load < time_left else time_left
            score += usable * 25.0 * p
            fleet.append((pos, load, p))

        bfs_data = self.dist_cache
        
        for p_pos, need in plants:
            if need <= 0: continue
            
            avg_rew = self.plant_values.get(p_pos, 0)
            if avg_rew == 0: continue
            
            dists_to_plant = bfs_data.get(p_pos)
            if not dists_to_plant:
                score -= 300 
                continue

            min_steps = 999
            best_potential = -1.0
            
            for r_pos, r_load, r_prob in fleet:
                if r_prob < 0.01: continue

                curr_steps = 999
                
                # --- BLOCKAGE CHECK (Crucial for correct scoring) ---
                if self.is_line_blocked(r_pos, p_pos, broken_locs):
                    curr_steps = 999
                else:
                    if r_load > 0:
                        curr_steps = dists_to_plant.get(r_pos, 999)
                    else:
                        if tap_positions:
                            for t_pos in tap_positions:
                                # Check if path to tap is blocked
                                if self.is_line_blocked(r_pos, t_pos, broken_locs): continue

                                d_plant_tap = dists_to_plant.get(t_pos, 999)
                                if d_plant_tap >= 999: continue

                                t_map = bfs_data.get(t_pos)
                                if not t_map: continue
                                d_robot_tap = t_map.get(r_pos, 999)
                                
                                total_d = d_robot_tap + d_plant_tap + 1 
                                if total_d < curr_steps:
                                    curr_steps = total_d

                if curr_steps < 999:
                    raw_val = avg_rew - (0.1 * curr_steps)
                    val_clipped = 1.0 if raw_val < 1.0 else raw_val
                    weighted_val = val_clipped * r_prob
                    
                    if curr_steps < min_steps:
                        min_steps = curr_steps
                    if weighted_val > best_potential:
                        best_potential = weighted_val
            
            if min_steps < (time_left * 1.5):
                if best_potential > 0:
                    score += (need * avg_rew) + best_potential
            else:
                score -= 300

        return score
    
    def prune_actions(self, state, legal_acts, time_left):
        """
        Filters out strategically poor actions. 
        REMOVED 'RESET' from the list to avoid wasting search budget.
        """
        if not legal_acts: return []
        
        final_list = []
        # NOTE: We purposely DO NOT add "RESET" here. 
        # Reset is handled by the "Smart Reset" logic at the root.
            
        robots, plants, taps, _ = state
        robot_map = {r[0]: (r[1], r[2]) for r in robots}
        broken_locs = {pos for rid, pos, _ in robots if self.robot_probs.get(rid, 1.0) < 0.01}

        grouped_moves = {}
        for action in legal_acts:
            if action == "RESET": continue
            parts = action.split()
            cmd = parts[0]
            rid = int(parts[1][1:-1])
            if rid not in grouped_moves: grouped_moves[rid] = []
            grouped_moves[rid].append((action, cmd))

        for rid, moves in grouped_moves.items():
            curr_pos, load = robot_map[rid]
            
            possible_targets = []
            if load > 0:
                possible_targets = [p[0] for p in plants if p[1] > 0]
            else:
                possible_targets = [t[0] for t in taps if t[1] > 0]

            valid_targets = []
            r_r, r_c = curr_pos
            
            # Simple line check for pruning
            for t_r, t_c in possible_targets:
                is_blocked = False
                if r_r == t_r: # Horizontal
                    min_c, max_c = min(r_c, t_c), max(r_c, t_c)
                    for br, bc in broken_locs:
                        if br == r_r and min_c < bc < max_c:
                            is_blocked = True
                            break
                elif r_c == t_c: # Vertical
                    min_r, max_r = min(r_r, t_r), max(r_r, t_r)
                    for br, bc in broken_locs:
                        if bc == r_c and min_r < br < max_r:
                            is_blocked = True
                            break
                
                if not is_blocked:
                    valid_targets.append((t_r, t_c))

            if not valid_targets:
                for act, cmd in moves:
                    if cmd != "LOAD": final_list.append(act)
                continue

            best_dist = 9999
            scored_moves = []
            
            for act, cmd in moves:
                dist_val = 9999
                
                if cmd == "POUR":
                    dist_val = -1 
                elif cmd == "LOAD":
                    if load < time_left: dist_val = -1
                    else: continue 
                else:
                    dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[cmd]
                    next_pos = (curr_pos[0]+dr, curr_pos[1]+dc)
                    
                    min_t_dist = 9999
                    for t in valid_targets:
                        t_map = self.dist_cache.get(t)
                        if t_map:
                            d = t_map.get(next_pos, 9999)
                            if d < min_t_dist: min_t_dist = d
                    dist_val = min_t_dist

                scored_moves.append((act, dist_val))
                if dist_val < best_dist:
                    best_dist = dist_val
            
            for act, val in scored_moves:
                if val <= best_dist:
                    final_list.append(act)

        return final_list
    
    def expectimax(self, state, depth, time_left):
        """
        Recursive search engine to find the max expected value.
        Optimized by inlining the transition logic to remove function call overhead.
        """
        # 1. Memoization Check (Cache Hit)
        # Using a tuple key is very fast in Python
        cache_id = (state, depth, time_left)
        cached_val = self.memo.get(cache_id)
        if cached_val is not None:
            return cached_val

        # 2. Base Cases
        _, _, _, total_need = state
        
        # VICTORY: All plants are watered
        # We add a huge bias (2000) to ensure the agent prefers finishing NOW 
        # rather than 5 steps later.
        if total_need == 0:
            return self.goal_reward + 2000.0

        # DEFEAT: Out of time
        if time_left <= 0:
            return -1000.0
            
        # CUTOFF: Depth limit reached, use Heuristic estimate
        if depth == 0:
            return self.heuristic(state, time_left)

        # 3. Get Candidates
        # We call our optimized pruning function immediately
        candidate_moves = self.get_legal_actions(state)
        if candidate_moves:
            candidate_moves = self.prune_actions(state, candidate_moves, time_left)
        
        # If stuck (no useful moves), fall back to heuristic
        if not candidate_moves:
            return self.heuristic(state, time_left)

        # 4. Maximization Loop (The "Max" Node)
        # We calculate the value of every move and pick the highest one.
        best_expected_value = -float('inf')
        
        # Optimization: Calculate recursion params once outside the loop
        next_depth = depth - 1
        next_time = time_left - 1
        
        for action in candidate_moves:
            # --- INLINED 'get_action_value' LOGIC ---
            # Calculating the expected value directly here saves 1 function call per branch.
            
            expected_score = 0.0
            outcomes = self.get_transitions(state, action)
            
            for next_state, prob, reward in outcomes:
                # RECURSION:
                # Value = Probability * ( Immediate_Reward + Future_Value )
                future_val = self.expectimax(next_state, next_depth, next_time)
                expected_score += prob * (reward + future_val)
            
            # Track the maximum
            if expected_score > best_expected_value:
                best_expected_value = expected_score
        
        # 5. Store in Cache and Return
        self.memo[cache_id] = best_expected_value
        return best_expected_value
    
    def choose_next_action(self, state):
        """
        Main interface. Decides depth dynamically and selects the best move.
        Optimized with a 'Single Candidate' fast-path to skip search when obvious.
        """
        # 1. Reset Cache
        self.memo.clear()
        
        # 2. Get Time and Step Data
        current_step = self.original_game.get_current_steps()
        time_left = self.horizon - current_step
        
        # 3. Smart Reset Check
        # We calculate the max potential of the current board.
        # If it's significantly worse than the starting board, we reset to try again.
        curr_val = self.get_best_route_val(state, time_left)
        
        # Threshold: -0.01 is a float safety margin
        if curr_val < (self.initial_best_route_val - 0.01):
            # Only reset if we have enough time (>5 steps) to actually improve
            if time_left > 5: 
                return "RESET"
        
        # 4. Generate Candidates
        legal_acts = self.get_legal_actions(state)
        if not legal_acts: return "RESET"

        # Prune dumb moves
        candidates = self.prune_actions(state, legal_acts, time_left)
        if not candidates: return "RESET"

        # --- OPTIMIZATION: Fast Path ---
        # If there is only 1 sane move, don't waste CPU searching. Just do it.
        if len(candidates) == 1:
            return candidates[0]

        # 5. Determine Dynamic Depth
        # We calculate how deep we can search without exceeding our node budget.
        # Logic: branching_factor ^ depth < budget
        branching_factor = len(candidates)
        node_budget = 6000 # Increased budget slightly as our search is now faster
        
        depth = 1
        current_nodes = branching_factor
        
        while depth < 30:
            next_layer = current_nodes * branching_factor
            if next_layer > node_budget:
                break
            current_nodes = next_layer
            depth += 1
            
        # 6. Run Search (Root Node)
        best_actions = []
        max_val = float('-inf')

        for action in candidates:
            # Calculate value for this specific root action
            # We pass depth, not depth-1, because get_action_value handles the decrement
            val = self.get_action_value(state, action, depth, time_left)
            
            # Update Best (handling float precision)
            if val > max_val:
                max_val = val
                best_actions = [action]
            elif abs(val - max_val) < 0.0001: # Float equality check
                best_actions.append(action)
        
        # 7. Final Selection
        if not best_actions: return "RESET"
        return random.choice(best_actions)