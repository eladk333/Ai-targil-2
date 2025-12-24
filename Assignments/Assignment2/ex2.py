import ext_plant
import random

# --- ENTER YOUR ID HERE ---
id = ["123456789"]

class Controller:
    """
    A Controller for the stochastic plant watering MDP.
    Uses Expectimax search with a heuristic that correctly values progress.
    """

    def __init__(self, game: ext_plant.Game):
        """Initialize controller for given game model."""
        self.game = game
        self.max_depth = 2  # Depth 2 is standard for this time limit
        
        # 1. Capture Static Problem Data
        self.problem = game.get_problem()
        self.rows, self.cols = self.problem["Size"]
        self.walls = set(self.problem.get("Walls", []))
        self.goal_reward = self.problem.get("goal_reward", 0)
        self.robot_probs = self.problem.get("robot_chosen_action_prob", {})
        self.capacities = game.get_capacities()
        
        # 2. Capture Initial State (Crucial for the "Reset" outcome)
        self.initial_state = game.get_current_state()
        
        # 3. Precompute Average Rewards for Plants
        self.avg_plant_rewards = {}
        for pos, rewards in self.problem.get("plants_reward", {}).items():
            if rewards:
                self.avg_plant_rewards[pos] = sum(rewards) / len(rewards)
            else:
                self.avg_plant_rewards[pos] = 0

    def choose_next_action(self, state):
        """
        Main entry point. Returns the best action as a string.
        """
        best_action = "RESET"
        best_value = -float('inf')
        
        # Get all legal actions for the current state
        legal_actions = self.get_legal_actions(state)
        
        if not legal_actions:
            return "RESET"

        # Run Expectimax for each top-level action
        for action in legal_actions:
            value = self.expect_node(state, action, depth=0)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action

    # --------------------------------------------------------------------------
    # Expectimax Logic
    # --------------------------------------------------------------------------

    def max_node(self, state, depth):
        if depth >= self.max_depth:
            return self.evaluate_state(state)

        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            return self.evaluate_state(state)

        max_val = -float('inf')
        for action in legal_actions:
            val = self.expect_node(state, action, depth)
            max_val = max(max_val, val)
        
        return max_val

    def expect_node(self, state, action, depth):
        transitions = self.get_transitions(state, action)
        
        expected_value = 0
        for prob, next_state, reward in transitions:
            # Value = Immediate Reward + Value of Future
            future_val = self.max_node(next_state, depth + 1)
            expected_value += prob * (reward + future_val)
            
        return expected_value

    # --------------------------------------------------------------------------
    # Heuristics & Evaluation (FIXED)
    # --------------------------------------------------------------------------

    def evaluate_state(self, state):
        """
        Heuristic function to estimate the value of a state.
        
        SCORING PHILOSOPHY:
        1. Remaining Need is a PENALTY (Negative). We want to minimize it (drive towards 0).
        2. Distance is a PENALTY. We want to be close to targets.
        3. Carrying Water is a small BONUS (to encourage loading), but smaller than the reward of pouring.
        """
        robots_t, plants_t, taps_t, total_water_need = state
        
        # If goal reached, this state is amazing.
        if total_water_need == 0:
            return self.goal_reward + 1000

        # 1. Calculate Penalty for Remaining Plant Needs
        # We multiply by Avg Reward so the agent treats "Need" as "Negative Reward Pending"
        need_penalty = 0
        plant_positions = {}
        for (pr, pc), need in plants_t:
            avg_r = self.avg_plant_rewards.get((pr, pc), 0)
            need_penalty += avg_r * need
            plant_positions[(pr, pc)] = need

        # 2. Calculate Distance Penalties and Load Bonuses
        distance_penalty = 0
        load_bonus = 0
        
        # Helper for Manhattan distance
        def dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        tap_positions = [pos for pos, water in taps_t if water > 0]
        
        for rid, (rr, rc), load in robots_t:
            robot_pos = (rr, rc)
            
            # Small bonus for having water (incentivize LOAD)
            # Must be smaller than the smallest plant reward to ensure POUR > HOLD
            load_bonus += (load * 0.5) 

            # Distance Logic
            if load > 0:
                # Robot has water -> Go to closest thirsty plant
                if plant_positions:
                    closest_dist = min(dist(robot_pos, p) for p in plant_positions)
                    distance_penalty += closest_dist * 0.2
            elif load < self.capacities[rid]:
                # Robot needs water -> Go to closest tap
                if tap_positions:
                    closest_dist = min(dist(robot_pos, t) for t in tap_positions)
                    distance_penalty += closest_dist * 0.2

        # Final Score:
        # We subtract need_penalty (because needs are bad).
        # We subtract distance_penalty (because travel is expensive).
        # We add load_bonus (because having water is useful).
        return -need_penalty - distance_penalty + load_bonus

    # --------------------------------------------------------------------------
    # Transition Model (Simulation)
    # --------------------------------------------------------------------------

    def get_transitions(self, state, action):
        """
        Returns a list of tuples: (probability, next_state, reward)
        """
        if action == "RESET":
            return [(1.0, self.initial_state, 0)]

        act_name, robot_id = self.parse_action(action)
        robots_t, _, _, _ = state
        
        # Locate robot
        robot_idx = -1
        r_pos = None
        for i, (rid, pos, _) in enumerate(robots_t):
            if rid == robot_id:
                robot_idx = i
                r_pos = pos
                break
        
        prob_success = self.robot_probs.get(robot_id, 1.0)
        outcomes = []
        
        # 1. Success Outcome
        ns_succ, rw_succ = self.apply_physics(state, robot_idx, act_name)
        outcomes.append((prob_success, ns_succ, rw_succ))
        
        # 2. Failure Outcomes
        if prob_success < 1.0:
            if act_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
                legal_moves = self.get_legal_moves_from_cell(r_pos, state, robot_id)
                possible_fails = [m for m in legal_moves if m != act_name] + ["STAY"]
                
                if not possible_fails: possible_fails = ["STAY"]

                fail_prob = (1.0 - prob_success) / len(possible_fails)
                for fail_move in possible_fails:
                    if fail_move == "STAY":
                        outcomes.append((fail_prob, state, 0))
                    else:
                        ns, nr = self.apply_physics(state, robot_idx, fail_move)
                        outcomes.append((fail_prob, ns, nr))
                        
            elif act_name == "POUR":
                ns, nr = self.apply_physics(state, robot_idx, "POUR_FAIL")
                outcomes.append((1.0 - prob_success, ns, 0))
                
            elif act_name == "LOAD":
                outcomes.append((1.0 - prob_success, state, 0))

        return outcomes

    def apply_physics(self, state, robot_idx, action):
        robots = list(state[0])
        plants = list(state[1])
        taps = list(state[2])
        total_need = state[3]
        
        rid, (rr, rc), load = robots[robot_idx]
        reward = 0
        
        if action == "POUR_FAIL":
            robots[robot_idx] = (rid, (rr, rc), max(0, load - 1))
            
        elif action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[action]
            robots[robot_idx] = (rid, (rr + dr, rc + dc), load)
            
        elif action == "LOAD":
            for i, (tpos, twater) in enumerate(taps):
                if tpos == (rr, rc):
                    new_twater = twater - 1
                    if new_twater > 0: taps[i] = (tpos, new_twater)
                    else: del taps[i]
                    break
            robots[robot_idx] = (rid, (rr, rc), load + 1)
            
        elif action == "POUR":
            for i, (ppos, pneed) in enumerate(plants):
                if ppos == (rr, rc):
                    new_pneed = pneed - 1
                    if new_pneed > 0: plants[i] = (ppos, new_pneed)
                    else: del plants[i]
                    total_need -= 1
                    reward += self.avg_plant_rewards.get(ppos, 0)
                    break
            robots[robot_idx] = (rid, (rr, rc), load - 1)
            
        # Check Global Goal (Reset)
        if total_need == 0:
            return self.initial_state, reward + self.goal_reward

        # Sort for canonical state (helps debugging and consistency)
        robots.sort(key=lambda x: x[0])
        plants.sort(key=lambda x: x[0])
        taps.sort(key=lambda x: x[0])
        
        return (tuple(robots), tuple(plants), tuple(taps), total_need), reward

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def get_legal_actions(self, state):
        actions = []
        robots_t, plants_t, taps_t, _ = state
        
        for i, (rid, (rr, rc), load) in enumerate(robots_t):
            other_robot_pos = {rpos for j, (_, rpos, _) in enumerate(robots_t) if i != j}
            
            # Moves
            for move, (dr, dc) in [("UP",(-1,0)), ("DOWN",(1,0)), ("LEFT",(0,-1)), ("RIGHT",(0,1))]:
                nr, nc = rr + dr, rc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in self.walls and (nr, nc) not in other_robot_pos:
                        actions.append(f"{move} ({rid})")
            
            # Interact
            if any(tpos == (rr, rc) for tpos, _ in taps_t) and load < self.capacities[rid]:
                actions.append(f"LOAD ({rid})")
            if any(ppos == (rr, rc) for ppos, _ in plants_t) and load > 0:
                actions.append(f"POUR ({rid})")
                
        return actions

    def get_legal_moves_from_cell(self, pos, state, my_rid):
        moves = []
        r, c = pos
        robots_t = state[0]
        other_robot_pos = {rpos for (rid, rpos, _) in robots_t if rid != my_rid}
        
        for move, (dr, dc) in [("UP",(-1,0)), ("DOWN",(1,0)), ("LEFT",(0,-1)), ("RIGHT",(0,1))]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if (nr, nc) not in self.walls and (nr, nc) not in other_robot_pos:
                    moves.append(move)
        return moves

    def parse_action(self, action_str):
        parts = action_str.split('(')
        return parts[0].strip(), int(parts[1].replace(')', '').strip())