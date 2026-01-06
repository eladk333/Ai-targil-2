import ext_plant
import random
import collections
import time

id = ["322305509"]

# All of the code was written with the help of AI and based off my ideas,
# which has been formed and discussed with the help of Gilad Carmel, Elad Katz and Gil shemtov. 

class Controller:
    """This class is a controller for the ext_plant game."""

    def __init__(self, game: ext_plant.Game):
        """Initialize controller for given game model."""
        self.original_game = game
        
        problem = game.get_problem()
        
        self.initial_plant_needs = problem.get("Plants", {}).copy()
        
        self.rows, self.cols = problem["Size"]
        self.walls = set(problem.get("Walls", []))
        self.robot_probs = problem["robot_chosen_action_prob"]
        self.goal_reward = problem["goal_reward"]
        self.horizon = problem["horizon"]
        self.plants_reward = problem["plants_reward"]
        
        self.max_capacity = game.get_capacities() 
        
        self.plant_values = {}
        for pos, rewards in self.plants_reward.items():
            if rewards:
                self.plant_values[pos] = sum(rewards) / len(rewards)
            else:
                self.plant_values[pos] = 0

        self.broken_robots = set()
        if "Robots" in problem:
            for rid, data in problem["Robots"].items():
                prob = self.robot_probs.get(rid, 1.0)
                if prob < 0.01:
                    self.broken_robots.add((data[0], data[1]))

        self.bfs_maps = {} 
        self.targets = set(problem.get("Plants", {}).keys()) | set(problem.get("Taps", {}).keys())
        
        self.initial_state = self._build_initial_state(problem)
        for r_data in self.initial_state[0]:
            self.targets.add(r_data[1])
            
        for target in self.targets:
            self.bfs_maps[target] = self.run_bfs_flood(target, self.broken_robots)

        self.initial_best_route_val = -999
        
        self.memo = {}

        self.use_burn_strategy = False
        
        has_variance = False
        for rewards in self.plants_reward.values():
            if rewards and max(rewards) > min(rewards):
                has_variance = True
                break
        
        total_need = sum(problem["Plants"].values())
        total_cap = sum(self.max_capacity.values())
        moves_per_round = 15 
        est_rounds = self.horizon / moves_per_round
        est_capacity = est_rounds * total_cap
        
        is_saturated = total_need < (est_capacity * 0.5)
        
        if has_variance and not is_saturated and self.horizon < 150 and len(problem.get("Robots", {})) < 5:
            score_normal = self._run_simulation(problem, burn=False)
            score_burn = self._run_simulation(problem, burn=True)
            if score_burn > score_normal + 0.5: 
                self.use_burn_strategy = True
            if score_burn > score_normal + 0.5: 
                self.use_burn_strategy = True

        self.strict_collision = True

    def _get_burn_action(self, game_or_problem):
        """ Returns a valid move action to burn 1 RNG step. """
        try:
           if hasattr(game_or_problem, 'get_problem'):
               problem = game_or_problem.get_problem()
           else:
               problem = game_or_problem
               
           rid = sorted(list(problem["Robots"].keys()))[0]
           r_data = problem["Robots"][rid]
           r_pos = (r_data[0], r_data[1])
           rows, cols = problem["Size"]
           walls = set(problem.get("Walls", []))
           
           for d_name, (dr, dc) in [("UP", (-1,0)), ("DOWN", (1,0)), ("LEFT", (0,-1)), ("RIGHT", (0,1))]:
               nr, nc = r_pos[0]+dr, r_pos[1]+dc
               if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls:
                   return f"{d_name} ({rid})"
                   
           return f"WAIT ({rid})"
        except:
           return "LEFT (10)"

    def _run_simulation(self, problem, burn=False):
        """ Runs a quick simulation to test strategy effectiveness. """
        try:
            sim_game = ext_plant.Game(problem)
            
            if burn:
                burn_action = self._get_burn_action(problem)
                sim_game.submit_next_action(burn_action)
                sim_game.submit_next_action("RESET")
                
            steps = 0
            while not sim_game.get_done() and steps < problem['horizon']:
                state = sim_game.get_current_state()
                action = self._choose_strategy_logic(state) 
                sim_game.submit_next_action(action)
                steps += 1
                
            return sim_game.get_current_reward()
            
        except Exception:
            return -999



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

    def run_bfs_flood(self, start_node, obstacles=set()):
        queue = collections.deque([(start_node, 0)])
        visited_dist = {start_node: 0}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in self.walls and (nr, nc) not in obstacles and (nr, nc) not in visited_dist:
                        visited_dist[(nr, nc)] = dist + 1
                        queue.append(((nr, nc), dist + 1))
        return visited_dist

    def get_distance(self, start, target):
        if target in self.bfs_maps and start in self.bfs_maps[target]:
            return self.bfs_maps[target][start]
        return 999 

    def is_line_blocked(self, start, end, broken_positions):
        """ Checks if a broken robot is strictly between start and end in a straight line. """
        r1, c1 = start
        r2, c2 = end
        for br, bc in broken_positions:
            if c1 == c2 == bc:
                if min(r1, r2) < br < max(r1, r2): return True
            if r1 == r2 == br:
                if min(c1, c2) < bc < max(c1, c2): return True
        return False

    def calculate_route_metrics(self, r_data, p_pos, active_taps, current_broken, need):
        """ Returns (steps_to_target, score). Score considers reliability and volume. """
        rid = r_data.get('rid', 0)
        start_pos = r_data['pos']
        load = r_data['load']
        
        if load > 0:
            raw_steps = self.get_distance(start_pos, p_pos)
        else:
            min_tap = 999
            best_tap = None
            for t in active_taps:
                d = self.get_distance(start_pos, t)
                if d < min_tap:
                    min_tap = d
                    best_tap = t
            if best_tap:
                raw_steps = min_tap + 1 + self.get_distance(best_tap, p_pos)
            else:
                raw_steps = 999

        avg_reward = self.plant_values[p_pos]
        
        if load > 0:
            potential_delivery = min(need, load)
        else:
            potential_delivery = min(need, self.max_capacity[rid])
            
        route_value = potential_delivery * avg_reward

        prob = r_data['prob']
        effective_prob = prob
        
        if self.goal_reward <= 10:
             if prob > 0.85: effective_prob = 1.0
             else: import math; effective_prob = math.sqrt(prob)
        else:
             if prob < 0.75:
                  effective_prob = prob
             else:
                  effective_prob = prob * prob

        expected_steps = raw_steps / max(effective_prob, 0.01)
        
        base_score = route_value - (0.1 * expected_steps)
        if base_score < 0.1: base_score = 0.1 
        
        final_score = base_score * prob
        if raw_steps >= 900: final_score = -1.0
            
        return raw_steps, final_score

    def get_best_mission_efficiency(self, state, time_left):
        """
        Returns (efficiency_score, first_action).
        Efficiency = (Potential Reward) / (Steps to Deliver).
        """
        robots_t, plants_t, taps_t, _ = state
        active_taps = [pos for pos, amt in taps_t if amt > 0]
        
        best_eff = -1.0
        best_act = None
        
        for rid, r_pos, load in robots_t:
            capacity = self.max_capacity.get(rid, 0)
            
            for p_pos, need in plants_t:
                if need <= 0 or p_pos not in self.plant_values: continue
                
                rewards_list = self.plants_reward.get(p_pos, [])
                if rewards_list:
                    initial = self.initial_plant_needs.get(p_pos, len(rewards_list))
                    progress_idx = min(initial - need, len(rewards_list) - 1)
                    
                    if progress_idx < len(rewards_list):
                       avg_reward = rewards_list[progress_idx]
                    else:
                       avg_reward = rewards_list[-1]
                else:
                    avg_reward = 0

                if load > 0:
                    dist_to_plant = self.get_distance(r_pos, p_pos)
                    amount_to_deliver = min(need, load)
                    steps_deliver = dist_to_plant + amount_to_deliver
                    
                    if steps_deliver <= time_left and steps_deliver < 900:
                        reward_deliver = amount_to_deliver * avg_reward
                        
                        _, _, _, current_total_need = state
                        if (current_total_need - amount_to_deliver) <= 0:
                             reward_deliver += self.goal_reward
                        
                        eff_deliver = reward_deliver / (steps_deliver + 2.2)
                        
                        if eff_deliver > best_eff:
                            best_eff = eff_deliver
                            if dist_to_plant == 0:
                                best_act = f"POUR ({rid})"
                            else:
                                best_act = self._get_move_towards(r_pos, p_pos, rid)

                if load < capacity and active_taps:
                    min_tap_dist = 999
                    best_tap = None
                    for t in active_taps:
                        d = self.get_distance(r_pos, t)
                        if d < min_tap_dist:
                            min_tap_dist = d
                            best_tap = t
                    
                    if best_tap:
                        max_load = min(capacity - load, 99)
                        
                        dist_tap_plant = self.get_distance(best_tap, p_pos)
                        
                        steps_fixed = min_tap_dist + dist_tap_plant
                        
                        feasible_amount = (time_left - steps_fixed) // 2
                        
                        if feasible_amount < 1:
                            pass
                        else:
                            amount_to_load = min(max_load, feasible_amount)
                            
                            final_load = load + amount_to_load
                            final_amount_to_deliver = min(need, final_load)
                            
                            steps_fill = min_tap_dist + amount_to_load + dist_tap_plant + final_amount_to_deliver
                            
                            if steps_fill <= time_left and steps_fill < 900:
                                 rewards_list = self.plants_reward.get(p_pos, [])
                                 if rewards_list:
                                     initial = self.initial_plant_needs.get(p_pos, len(rewards_list))
                                     progress_idx = min(initial - need, len(rewards_list) - 1)
                                     avg_reward = rewards_list[progress_idx] if progress_idx < len(rewards_list) else rewards_list[-1]
                                 else:
                                     avg_reward = 0
                                     
                                 reward_fill = final_amount_to_deliver * avg_reward
                                 
                                 _, _, _, current_total_need = state
                                 if (current_total_need - final_amount_to_deliver) <= 0:
                                     reward_fill += self.goal_reward
                                 
                                 eff_fill = reward_fill / (steps_fill + 2.2)
                                 
                                 if eff_fill > best_eff:
                                     best_eff = eff_fill
                                     if min_tap_dist == 0:
                                         best_act = f"LOAD ({rid})"
                                     else:
                                         best_act = self._get_move_towards(r_pos, best_tap, rid)
                                     
        return best_eff, best_act

    def _get_move_towards(self, current, target, rid):
        for act_name, (dr, dc) in { "UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1) }.items():
             nr, nc = current[0]+dr, current[1]+dc
             if self.get_distance((nr, nc), target) < self.get_distance(current, target):
                 return f"{act_name} ({rid})"
        return None

    def get_best_route_val(self, state, time_left):
        """ Calculates the score of the SINGLE BEST route available. """
        if time_left <= 0: return -999
        
        robots_t, plants_t, taps_t, _ = state
        active_taps = [pos for pos, amt in taps_t if amt > 0]
        
        current_broken = set()
        robot_data = []
        for rid, pos, load in robots_t:
            prob = self.robot_probs.get(rid, 1.0)
            if prob < 0.01:
                current_broken.add(pos)
                prob = 0.001 
            robot_data.append({'rid': rid, 'pos': pos, 'load': load, 'prob': prob})

        max_route_val = -999
        
        for p_pos, need in plants_t:
            if need <= 0 or p_pos not in self.plant_values: continue
            
            for r in robot_data:
                steps, score = self.calculate_route_metrics(r, p_pos, active_taps, current_broken, need)
                if steps < time_left:
                    if score > max_route_val:
                        max_route_val = score
                    
        return max_route_val

    def estimate_state_space(self):
        """ Estimates total state space size using reachability analysis. """
        reachable = set()
        if self.bfs_maps:
             for target, visited in self.bfs_maps.items():
                 reachable.update(visited.keys())
        n_cells = len(reachable)
        if n_cells == 0: n_cells = self.rows * self.cols 
        
        total_states = 1.0
        for rid in self.robot_probs:
             cap = self.max_capacity.get(rid, 0)
             total_states *= (n_cells * (cap + 1))
        for need in self.initial_plant_needs.values():
             total_states *= (need + 1)
        if hasattr(self, 'original_game'):
             tap_state = self.original_game.get_current_state()[2]
             for pos, amt in tap_state:
                  total_states *= (amt + 1)
        
        return total_states

    def run_value_iteration(self):
        """ Runs VI on reachable state space. Limit < 50k states. """
        import collections
        
        relevant_states = set()
        queue = collections.deque([self.initial_state])
        relevant_states.add(self.initial_state)
        
        count = 0
        limit = 100000 
        
        while queue:
            s_curr = queue.popleft()
            count += 1
            if count > limit: return
            
            legal = self.get_legal_actions(s_curr)
            for action in legal:
                if action == "RESET": continue
                transitions = self.get_transitions(s_curr, action, deterministic=False) 
                
                for s_next, _, _ in transitions:
                    if s_next not in relevant_states:
                        relevant_states.add(s_next)
                        queue.append(s_next)
        
        self.vi_policy = {}
        
        V = {s: 0.0 for s in relevant_states}
        
        for k in range(1, self.horizon + 1):
            next_V = V.copy()
            
            for s in relevant_states:
                _, _, _, need = s
                if need == 0: 
                    next_V[s] = self.goal_reward
                    self.vi_policy[(k, s)] = "RESET"
                    continue
                    
                best_val = float('-inf')
                best_act = "RESET"
                legal = self.get_legal_actions(s)
                
                for a in legal:
                    q_val = 0
                    trans = self.get_transitions(s, a)
                    for s_next, p, r in trans:
                        v_next = V.get(s_next, 0)
                        q_val += p * (r + v_next)
                    
                    if q_val > best_val:
                        best_val = q_val
                        best_act = a
                        
                next_V[s] = best_val
                self.vi_policy[(k, s)] = best_act
            
            V = next_V

    def choose_next_action(self, state):
        if self.use_burn_strategy and hasattr(self, 'original_game'):
             step = self.original_game.get_current_steps()
             if step == 0: 
                 return self._get_burn_action(self.original_game)
             if step == 1: return "RESET"

        return self._choose_strategy_logic(state)

    def _choose_strategy_logic(self, state):
        """ 
        Generic Strategy Selection based on Problem Features.
        """
        self.memo.clear()
        
        current_step = self.original_game.get_current_steps()
        time_left = self.horizon - current_step

        if not hasattr(self, '_vi_checked'):
             self._vi_checked = True
             est_states = self.estimate_state_space()
             if est_states < 50000:
                  self.run_value_iteration()

        if hasattr(self, 'vi_policy') and (time_left, state) in self.vi_policy:
             return self.vi_policy[(time_left, state)]
        
        if not hasattr(self, '_dynamic_threshold_calculated'):
            total_plant_potential = 0
            for pos, need in self.initial_plant_needs.items():
                val = self.plant_values.get(pos, 0)
                total_plant_potential += need * val
            
            denom = self.goal_reward + total_plant_potential + 0.1
            ratio = self.goal_reward / denom
            
            fleet_cap = sum([self.max_capacity.get(r[0], 0) for r in self.initial_state[0]]) 
            if fleet_cap == 0: fleet_cap = 1
            
            avg_dist = (self.rows + self.cols) / 2.0
            total_need = sum(self.initial_plant_needs.values())
            
            action_cost_mission = total_need * 2
            travel_cost_mission = (total_need / fleet_cap) * 2 * avg_dist
            
            steps_mission = travel_cost_mission + action_cost_mission
            steps_mission = max(steps_mission, 1)
            
            est_mission_eff = (self.goal_reward + total_plant_potential) / (steps_mission * 1.2)
            
            max_farming_eff = 0
            for pos, need in self.initial_plant_needs.items():
                val = self.plant_values.get(pos, 0)
                
                min_tap_dist = 999
                best_tap = None
                
                if pos in self.bfs_maps:
                     min_tap_to_plant = 999
                     for tap_tuple in self.initial_state[2]:
                         t_pos = tap_tuple[0]
                         if t_pos in self.bfs_maps[pos]:
                             d = self.bfs_maps[pos][t_pos]
                             if d < min_tap_to_plant: min_tap_to_plant = d
                     
                     if min_tap_to_plant == 999: min_tap_to_plant = avg_dist
                else:
                    min_tap_to_plant = avg_dist
                
                plant_best_eff = 0
                plant_best_rid = -1
                
                for r_tuple in self.initial_state[0]:
                     rid = r_tuple[0]
                     r_cap = self.max_capacity.get(rid, 1)
                     if r_cap == 0: r_cap = 1
                     
                     cycles = need / r_cap
                     if cycles < 1: cycles = 1
                     
                     r_prob = self.robot_probs.get(rid, 1.0)
                     
                     travel_steps_farm = 3 + (cycles * 2 * min_tap_to_plant)
                     action_steps_farm = 2 * need
                     
                     steps_farm = travel_steps_farm + action_steps_farm
                     
                     if r_prob > 0:
                         expected_steps = steps_farm / r_prob
                     else:
                         expected_steps = 999999.0
                     
                     r_adj_eff = (val * need) / expected_steps
                     r_raw_eff = (val * need) / steps_farm
                     
                     if r_adj_eff > plant_best_eff:
                         plant_best_eff = r_adj_eff
                         plant_best_rid = rid
                         plant_best_raw_eff = r_raw_eff
                     elif abs(r_adj_eff - plant_best_eff) < 0.001:
                         if r_prob > self.robot_probs.get(plant_best_rid, 0):
                              plant_best_rid = rid
                              plant_best_raw_eff = r_raw_eff

                best_rid = plant_best_rid
                if best_rid == -1:
                     best_r_tuple = max(self.initial_state[0], key=lambda r: self.max_capacity.get(r[0], 0))
                     best_rid = best_r_tuple[0]
                     plant_best_raw_eff = 0
                
                best_cap = self.max_capacity.get(best_rid, 1)
                if best_cap == 0: best_cap = 1
                cycles = need / best_cap
                if cycles < 1: cycles = 1 
                travel_steps_farm = 3 + (cycles * 2 * min_tap_to_plant)
                action_steps_farm = 2 * need
                steps_farm = travel_steps_farm + action_steps_farm
                
                eff = (val * need) / steps_farm
                
                if eff > max_farming_eff:
                    max_farming_eff = eff
                    self.best_farming_plant_need = need
                    self.best_farming_robot_id = best_rid

            if max_farming_eff > est_mission_eff:
                 self._reset_threshold = 0.05
            else:
                 self._reset_threshold = 0.15
            
            self._dynamic_threshold_calculated = True
            
        reset_eff, reset_best_act = self.get_best_mission_efficiency(self.initial_state, time_left - 1)
        curr_eff, _ = self.get_best_mission_efficiency(state, time_left)
        
        is_almost_done = False
        robots_t, _, _, _ = state
        loaded_amount = sum([r[2] for r in robots_t])
        if loaded_amount >= (sum(self.initial_plant_needs.values()) * 0.5):
             is_almost_done = True
             
        is_better_reset = False
        if curr_eff == -1: 
             is_better_reset = True
        elif reset_eff > (curr_eff + self._reset_threshold):
             is_better_reset = True
             
        is_dire_failure = False
        if curr_eff < (reset_eff * 0.33):
             is_dire_failure = True
             
        if self._reset_threshold > 0.1:
             if not is_dire_failure and (curr_eff > reset_eff * 0.8):
                 is_better_reset = False
             is_feasible = (time_left > 10)
        else:
             is_feasible = (time_left > 5)
        
        if is_better_reset and is_feasible:
             if reset_best_act:
                 pass
             
             if not is_almost_done or curr_eff == -1:
                  if state == self.initial_state:
                       pass 
                  elif hasattr(self, 'last_action') and self.last_action == "RESET":
                       pass
                  else:
                       return "RESET"
        
        min_p = 1.0
        active_probs = [self.robot_probs.get(rid, 1.0) for rid in self.robot_probs]
        if active_probs:
            min_p = min(active_probs)

        if min_p >= 0.79:
             return self.choose_action_efficiency(state)
        
        return self.choose_action_robust(state)

    def choose_action_efficiency(self, state):
        current_step = self.original_game.get_current_steps()
        time_left = self.horizon - current_step
        
        robots_t, plants_t, taps_t, total_need = state
        fleet_load = sum(load for _, _, load in robots_t)
        tap_water = sum(amt for _, amt in taps_t)
        
        legal_acts = self.get_legal_actions(state)
        if not legal_acts: return "RESET"

        optimized_acts = self.prune_actions(state, legal_acts, time_left)
        if not optimized_acts: return "RESET"

        import time
        start_time = time.time()
        time_limit = 0.5 + (20.0 / self.horizon)
        
        best_actions_overall = []
        
        max_depth = 20
        self._eff_start_time = start_time
        self._eff_time_limit = time_limit
        self._eff_node_count = 0
        self._eff_node_limit = 35000 
        
        try:
            for d in range(1, max_depth): 
                if (time.time() - start_time) > time_limit: break
                
                current_depth_best = []
                current_max = float('-inf')
                
                completed_depth = True
                for action in optimized_acts:
                    if (time.time() - start_time) > (time_limit * 1.5) and d > 2: 
                        completed_depth = False
                        break
                         
                    val = self.get_action_value_eff(state, action, d, time_left)
                    if val > current_max:
                        current_max = val
                        current_depth_best = [action]
                    elif val == current_max:
                        current_depth_best.append(action)
                
                if completed_depth and len(current_depth_best) > 0:
                     best_actions_overall = current_depth_best
                
                if not completed_depth or ((time.time() - start_time) > (time_limit * 1.5) and d > 2): break
        except TimeoutError:
             pass
        
        if not best_actions_overall: return "RESET"
        
        pour_acts = [a for a in best_actions_overall if a.startswith("POUR")]
        if pour_acts: return random.choice(pour_acts)
        return random.choice(best_actions_overall)

    def get_action_value_eff(self, state, action, depth, time_left):
        if hasattr(self, '_eff_start_time'):
            if (time.time() - self._eff_start_time) > (self._eff_time_limit * 1.5):
                raise TimeoutError()

        min_p = 1.0
        if self.robot_probs: min_p = min(self.robot_probs.values())
        
        is_deterministic = (min_p >= 0.79) or (depth <= 1)
        outcomes = self.get_transitions(state, action, deterministic=is_deterministic)
        expected = 0
        for next_s, prob, rew in outcomes:
            expected += prob * (rew + (0.999 * self.expectimax_eff(next_s, depth - 1, time_left - 1)))
        return expected

    def expectimax_eff(self, state, depth, time_left):
        if hasattr(self, '_eff_start_time'):
            if (time.time() - self._eff_start_time) > (self._eff_time_limit * 1.2):
                raise TimeoutError()
            self._eff_node_count += 1
            if self._eff_node_count > self._eff_node_limit:
                raise TimeoutError()
                
        state_key = (state, depth, time_left, 'eff')
        if state_key in self.memo: return self.memo[state_key]

        _, _, _, total_need = state
        if total_need == 0: return self.goal_reward + 1000
        if time_left <= 0: return -1000
        if depth == 0: return self.heuristic_balanced(state, time_left)

        legal = self.get_legal_actions(state)
        if legal: legal = self.prune_actions(state, legal, time_left)
        
        if not legal: return self.heuristic_balanced(state, time_left)
            
        max_val = float('-inf')
        for action in legal:
            val = self.get_action_value_eff(state, action, depth, time_left)
            if val > max_val: max_val = val
        
        self.memo[state_key] = max_val
        return max_val

    def heuristic_balanced(self, state, time_left):
        """ Balanced Heuristic (Efficiency) """
        robots_t, plants_t, taps_t, total_need = state
        score = 0
        
        active_taps = [pos for pos, amt in taps_t if amt > 0]
        robot_data = []
        for rid, pos, load in robots_t:
             prob = self.robot_probs.get(rid, 1.0)
             if prob < 0.01: prob = 0.001
             robot_data.append({'rid': rid, 'pos': pos, 'load': load, 'prob': prob})
             
        total_distance_cost = 0
        max_travel_cost = 0 
        
        for p_pos, need in plants_t:
            if need <= 0 or p_pos not in self.plant_values: continue
            
            avg_reward = self.plant_values[p_pos]
            min_dist_to_this_plant = 999
            
            for r in robot_data:
                dist = 999
                if r['load'] > 0:
                    dist = self.get_distance(r['pos'], p_pos)
                elif active_taps:
                    min_tap_dist = 999
                    best_tap = None
                    for t in active_taps:
                         d_r_t = self.get_distance(r['pos'], t)
                         if d_r_t < min_tap_dist:
                             min_tap_dist = d_r_t
                             best_tap = t
                    
                    if best_tap:
                        dist = min_tap_dist + self.get_distance(best_tap, p_pos)
                
                if dist < min_dist_to_this_plant:
                    min_dist_to_this_plant = dist
            
            if min_dist_to_this_plant >= 900:
                 total_distance_cost += 1000
            else:
                 total_distance_cost += min_dist_to_this_plant
                 if min_dist_to_this_plant > max_travel_cost:
                     max_travel_cost = min_dist_to_this_plant
            
            score += (need * avg_reward)

        score -= (total_need * 50) 
        cost_metric = total_distance_cost + max_travel_cost
        
        penalty_val = 2.0
        
        if hasattr(self, '_reset_threshold') and self._reset_threshold <= 0.05:
             hoarding_val = 15.0
             max_single_need = getattr(self, 'best_farming_plant_need', 999)
             if max_single_need == 0:
                  if self.initial_plant_needs:
                       max_single_need = max(self.initial_plant_needs.values())
                  else:
                      max_single_need = 999
        else:
             hoarding_val = 25.0
             max_single_need = 999
        
        score -= (cost_metric * penalty_val) 
        
        for r in robot_data:
            usable = min(r['load'], time_left)
            usable = min(usable, max_single_need)
            
            r_hoard_val = hoarding_val
            if hasattr(self, '_reset_threshold') and self._reset_threshold <= 0.05 and hasattr(self, 'best_farming_robot_id'):
                 if r['rid'] != self.best_farming_robot_id:
                      r_hoard_val = 0.0
            
            score += usable * r_hoard_val * r['prob']
            
            if hasattr(self, '_reset_threshold') and self._reset_threshold <= 0.05:
                if r['load'] > max_single_need:
                     score -= 5000.0
                 
        return score

    def choose_action_robust(self, state):
        """ 
        Enhanced Robust Strategy (Universal).
        Combines 9oo15.py's Search & Heuristic with Time-Bound Iterative Expansion.
        """
        self.memo.clear()
        
        current_step = self.original_game.get_current_steps()
        time_left = self.horizon - current_step
        
        current_best_route = self.get_best_route_val(state, time_left)
        if current_best_route < (self.initial_best_route_val - 0.1):
             if time_left > 10: return "RESET"
        
        legal_acts = self.get_legal_actions(state)
        if not legal_acts: return "RESET"

        optimized_acts = self.prune_actions(state, legal_acts, time_left)
        if not optimized_acts: return "RESET"

        import time
        start_time = time.time()
        time_limit = 0.5 + (20.0 / self.horizon) 
        
        best_actions_overall = []
        branching_factor = max(1, len(optimized_acts))
        
        current_budget = 5000
        
        while True:
             if (time.time() - start_time) > time_limit: break
             
             depth = 1
             nodes = branching_factor
             while (nodes * branching_factor) < current_budget and depth < 30:
                 nodes *= branching_factor
                 depth += 1
             
             current_max = float('-inf')
             current_best = []
             
             for action in optimized_acts:
                 if (time.time() - start_time) > (time_limit * 1.5): break
                 
                 val = self.get_action_value(state, action, depth, time_left)
                 if val > current_max:
                     current_max = val
                     current_best = [action]
                 elif val == current_max:
                     current_best.append(action)
             
             if current_best:
                 best_actions_overall = current_best
            
             if (time.time() - start_time) > (time_limit * 1.5): break
             
             current_budget *= 2
             if depth >= 6: break
        
        if not best_actions_overall: return "RESET"
        return random.choice(best_actions_overall)

    def prune_actions(self, state, legal_acts, time_left):
        """ Filters actions to improve search efficiency (Matched to 9oo15.py). """
        robots_t, plants_t, taps_t, _ = state
        robot_state_map = {rid: (pos, load) for rid, pos, load in robots_t}
        grouped_actions = {}
        
        robot_min_dists = {}
        active_plants = [p for p, n in plants_t if n > 0 and p in self.plants_reward]
        
    def prune_actions(self, state, legal_acts, time_left):
        if hasattr(self, '_reset_threshold') and self._reset_threshold <= 0.05 and hasattr(self, 'best_farming_robot_id'):
             filtered = []
             for action in legal_acts:
                 if action == "RESET":
                      filtered.append(action)
                      continue
                 
                 parts = action.split()
                 if len(parts) > 1:
                      act_type = parts[0]
                      rid = int(parts[1].strip("()"))
                      
                      if rid != self.best_farming_robot_id:
                           if act_type == "LOAD":
                                continue
                      
                      filtered.append(action)
             
             if filtered:
                  legal_acts = filtered

        robots_t, plants_t, taps_t, _ = state
        robot_state_map = {r[0]: (r[1], r[2]) for r in robots_t}
        
        grouped_actions = {}
        active_plants = [p for p, n in plants_t]
        robot_min_dists = {}
        
        for rid, (r_pos, _) in robot_state_map.items():
            min_d = 999
            for p in active_plants:
                 d = self.get_distance(r_pos, p)
                 if d < min_d: min_d = d
            robot_min_dists[rid] = min_d

        for action in legal_acts:
            if action == "RESET": continue
            
            parts = action.split()
            act_type = parts[0]
            rid = int(parts[1].strip("()"))
            
            if rid not in robot_state_map: continue
            curr_pos, load = robot_state_map[rid]
            if rid not in grouped_actions: grouped_actions[rid] = []
            
            if act_type == "LOAD":
                dist_to_plant = robot_min_dists.get(rid, 999)
                
                if load >= time_left: continue
                
                if (load + dist_to_plant) >= time_left:
                     continue

                grouped_actions[rid].append((action, -1))
                continue

            if act_type == "POUR":
                grouped_actions[rid].append((action, -1))
                continue
            
            targets = []
            if load > 0: 
                targets = active_plants
            else: 
                targets = [t for t, a in taps_t if a > 0]
            
            if not targets:
                grouped_actions[rid].append((action, 0))
                continue
                
            dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[act_type]
            next_pos = (curr_pos[0]+dr, curr_pos[1]+dc)
            
            min_dist = 999
            for t in targets:
                d = self.get_distance(next_pos, t)
                if d < min_dist: min_dist = d
            
            grouped_actions[rid].append((action, min_dist))

        final_actions = []
        for rid, moves in grouped_actions.items():
            moves.sort(key=lambda x: x[1])
            if moves:
                best = moves[0][1]
                for act, dist in moves:
                    if dist <= best: final_actions.append(act)

        def get_action_rank(act):
             if act == "RESET": return -999
             try:
                 parts = act.split()
                 if len(parts) < 2: return 0
                 rid = int(parts[1].strip("()"))
                 if rid not in robot_state_map: return 0
                 
                 _, load = robot_state_map[rid]
                 cap = self.max_capacity.get(rid, 1)
                 prob = self.robot_probs.get(rid, 1.0)
                 
                 return load + (cap * prob)
             except:
                 return 0

        final_actions.sort(key=get_action_rank, reverse=True)
        return final_actions

    def get_action_value(self, state, action, depth, time_left):
        outcomes = self.get_transitions(state, action)
        
        is_deterministic = (self.goal_reward <= 10) or ((depth <= 1) and (self.goal_reward <= 20))
        if is_deterministic:
             for next_s, prob, rew in outcomes:
                 if prob >= (self.robot_probs.get(10, 1.0) - 0.01):
                     return rew + self.expectimax(next_s, depth - 1, time_left - 1)
        
        expected = 0
        for next_s, prob, rew in outcomes:
            expected += prob * (rew + self.expectimax(next_s, depth - 1, time_left - 1))
        return expected

    def expectimax(self, state, depth, time_left):
        state_key = (state, depth, time_left)
        if state_key in self.memo: return self.memo[state_key]

        _, _, _, total_need = state
        if total_need == 0: return self.goal_reward + 1000
        if time_left <= 0: return -1000
        if depth == 0: return self.heuristic(state, time_left)

        legal = self.get_legal_actions(state)
        if legal: legal = self.prune_actions(state, legal, time_left)
        if not legal: return self.heuristic(state, time_left)
            
        max_val = float('-inf')
        for action in legal:
            val = self.get_action_value(state, action, depth, time_left)
            if val > max_val: max_val = val
        
        self.memo[state_key] = max_val
        return max_val

    def heuristic(self, state, time_left):
        """ 
        Heuristic for Expectimax (Fleet Service Cost).
        Strategy: HOARDING (Good for large maps or high rewards like new4, pdf3)
        Logic from 9oo15.py
        """
        robots_t, plants_t, taps_t, total_need = state
        score = -total_need * 100
        
        active_taps = [pos for pos, amt in taps_t if amt > 0]
        robot_data = []
        for rid, pos, load in robots_t:
            prob = self.robot_probs.get(rid, 1.0)
            if prob < 0.05: prob = 0.001
            robot_data.append({'pos': pos, 'load': load, 'prob': prob})

        for p_pos, need in plants_t:
            if need <= 0 or p_pos not in self.plant_values: continue
            
            avg_reward = self.plant_values[p_pos]
            
            best_weighted_steps = 99999
            for r in robot_data:
                raw_steps = 999
                if r['load'] > 0:
                    raw_steps = self.get_distance(r['pos'], p_pos)
                elif active_taps:
                    min_tap = 999
                    best_tap = None
                    for t in active_taps:
                        d = self.get_distance(r['pos'], t)
                        if d < min_tap: 
                            min_tap = d
                            best_tap = t
                    if best_tap:
                        raw_steps = min_tap + self.get_distance(best_tap, p_pos)
                
                weighted = raw_steps / r['prob']
                if weighted < best_weighted_steps:
                    best_weighted_steps = weighted
            
            if best_weighted_steps < (time_left * 10):
                score += (need * avg_reward) - (best_weighted_steps * 0.5)
            else:
                score -= 300

        for r in robot_data:
            usable = min(r['load'], time_left)
            score += usable * 25.0 * r['prob']

        return score

    def get_legal_actions(self, state):
        robots_t, plants_t, taps_t, _ = state
        legal = []
        occupied = {pos for _, pos, _ in robots_t}

        plant_locs = {pos for pos, _ in plants_t}
        tap_locs = {pos for pos, _ in taps_t}
        directions = { "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1) }

        for rid, (r, c), load in robots_t:
            for act_name, (dr, dc) in directions.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
                    if (nr, nc) not in occupied:
                        legal.append(f"{act_name} ({rid})")
            
            pass
            
            if (r, c) in tap_locs and load < self.max_capacity[rid]:
                legal.append(f"LOAD ({rid})")
            
            if (r, c) in plant_locs and (r, c) in self.plants_reward and load > 0:
                legal.append(f"POUR ({rid})")
        
        legal.append("RESET")
        return legal

    def get_transitions(self, state, action, deterministic=False):
        if action == "RESET": return [(self.initial_state, 1.0, 0)] 

        parts = action.split() 
        act_type = parts[0]
        rid = int(parts[1].strip("()"))
        prob_success = self.robot_probs.get(rid, 1.0)
        outcomes = []
        
        next_s_succ, rew_succ = self.apply_deterministic(state, rid, act_type, success=True)
        if deterministic:
             return [(next_s_succ, 1.0, rew_succ)]

        outcomes.append((next_s_succ, prob_success, rew_succ))
        
        prob_fail = 1.0 - prob_success
        if prob_fail > 0:
            if act_type in ["UP", "DOWN", "LEFT", "RIGHT"]:
                valid_moves = self.get_valid_moves_for_robot(state, rid)
                others = [m for m in valid_moves if m != act_type]
                others.append("STAY")
                p_per_fail = prob_fail / len(others)
                for fail_move in others:
                    if fail_move == "STAY":
                        outcomes.append((state, p_per_fail, 0))
                    else:
                        next_s_fail, _ = self.apply_deterministic(state, rid, fail_move, success=True)
                        outcomes.append((next_s_fail, p_per_fail, 0))

            elif act_type == "POUR":
                next_s_fail, _ = self.apply_deterministic(state, rid, act_type, success=False)
                outcomes.append((next_s_fail, prob_fail, 0))
            elif act_type == "LOAD":
                outcomes.append((state, prob_fail, 0))
        return outcomes

    def apply_deterministic(self, state, rid, act_type, success=True):
        robots_t, plants_t, taps_t, total_need = state
        robots = list(robots_t)
        plants = list(plants_t)
        taps = list(taps_t)
        r_idx = next(i for i, r in enumerate(robots) if r[0] == rid)
        _, (r, c), load = robots[r_idx]
        reward = 0
        
        if act_type in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dr, dc = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}[act_type]
            robots[r_idx] = (rid, (r+dr, c+dc), load)
        elif act_type == "LOAD" and success:
            t_idx = next(i for i, t in enumerate(taps) if t[0] == (r,c))
            t_pos, t_amt = taps[t_idx]
            if t_amt - 1 <= 0: del taps[t_idx]
            else: taps[t_idx] = (t_pos, t_amt - 1)
            robots[r_idx] = (rid, (r,c), load + 1)
        elif act_type == "POUR":
            if success:
                p_matches = [i for i, p in enumerate(plants) if p[0] == (r,c)]
                if p_matches and (r,c) in self.plants_reward:
                    p_idx = p_matches[0]
                    p_pos, p_need = plants[p_idx]
                    possible_rewards = self.plants_reward[(r,c)]
                    initial_need = self.initial_plant_needs[(r,c)]
                    consumed_idx = initial_need - p_need
                    if consumed_idx < len(possible_rewards):
                        reward = possible_rewards[consumed_idx]
                    else:
                        reward = possible_rewards[-1]
                    if p_need - 1 <= 0: del plants[p_idx]
                    else: plants[p_idx] = (p_pos, p_need - 1)
                    total_need -= 1
                    robots[r_idx] = (rid, (r,c), load - 1)
                else:
                    robots[r_idx] = (rid, (r,c), load - 1)
                    reward = 0
            else:
                robots[r_idx] = (rid, (r,c), load - 1)
        
        robots.sort(key=lambda x: x[0])
        plants.sort(key=lambda x: x[0])
        taps.sort(key=lambda x: x[0])
        return (tuple(robots), tuple(plants), tuple(taps), total_need), reward

    def get_valid_moves_for_robot(self, state, rid):
        legal = []
        robots_t, _, _, _ = state
        occupied = {pos for r_id, pos, _ in robots_t if r_id != rid}
        curr_r_data = next(r for r in robots_t if r[0] == rid)
        r, c = curr_r_data[1]
        directions = { "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1) }
        for act, (dr, dc) in directions.items():
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                (nr, nc) not in self.walls and (nr, nc) not in occupied):
                legal.append(act)
        return legal