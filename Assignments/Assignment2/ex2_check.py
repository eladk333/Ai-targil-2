import ext_plant
import ex2#_ori as ex2
import numpy as np
import time



def solve(game: ext_plant.Game):
    policy = ex2.Controller(game)
    for i in range(game.get_max_steps()):
        game.submit_next_action(chosen_action=policy.choose_next_action(game.get_current_state()))
        if game.get_done():
            break
    print('Game result:', game.get_current_state(), '\n\tFinished in', game.get_max_steps(),
         'Steps.\n\tReward result->',game.get_current_reward())
    print("Game finished ", "" if game.get_current_state()[-1] else "un", "successfully.", sep='')
    game.show_history()
    return game.get_current_reward()

problem_pdf = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.9,
    },
    "goal_reward": 10,
    "plants_reward": {
        (0, 2) : [1,2,3,4],
        (2, 0) : [1,2,3,4],
    },
    "seed": 45,
    "horizon": 30,
}

problem_pdf2 = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob":{
        10: 0.9,
        11: 0.8,
    },
    "goal_reward": 12,
    "plants_reward": {
        (0, 2) : [1,3,5,7],
        (2, 0) : [1,2,3,4],
    },
    "seed": 45,
    "horizon": 35,
}

problem_pdf3 = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob":{
        10: 0.7,
        11: 0.6,
    },
    "goal_reward": 30,
    "plants_reward": {
        (0, 2) : [1,2,3,4],
        (2, 0) : [10,11,12,13],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new1_version1 = {
    "Size":  (5, 6),

    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2), (1, 3),
        (3, 2), (3, 3),
    },

    "Taps": {
        (2, 2): 12,
    },

    "Plants": {
        (0, 1): 3,
        (4, 5): 6,
    },

    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
     "robot_chosen_action_prob":{
        10: 0.9,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5) : [1,2,3,4],
        (0, 1) : [10,11,12,13],
    },
    "seed": 45,
    "horizon": 30,
}
problem_new1_version2 = {
    "Size":  (5, 6),

    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2), (1, 3),
        (3, 2), (3, 3),
    },

    "Taps": {
        (2, 2): 12,
    },

    "Plants": {
        (0, 1): 3,
        (4, 5): 6,
    },

    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
     "robot_chosen_action_prob":{
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5) : [1,2,3,4],
        (0, 1) : [10,11,12,13],
    },
    "seed": 45,
    "horizon": 70,
}
problem_new1_version3 = {
    "Size":  (5, 6),

    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2), (1, 3),
        (3, 2), (3, 3),
    },

    "Taps": {
        (2, 2): 12,
    },

    "Plants": {
        (0, 1): 2,
        (4, 5): 6,
    },

    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
     "robot_chosen_action_prob":{
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5) : [1,2,3,4],
        (0, 1) : [10,11,12,13],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new2_version1 = {
    "Size":  (5, 6),

    "Walls": {
        # corridor shifted up
        (0, 2), (0, 3),
        (2, 2), (2, 3),
    },

    "Taps": {
        (1, 2): 10,         # upper tap
        (3, 3): 10,         # lower tap
    },

    "Plants": {
        (0, 0): 5,         # top-left
        (4, 5): 5,         # bottom-right
    },

    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0) : [5,7],
        (4, 5) : [5,7],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new2_version2 = {
    "Size":  (5, 6),

    "Walls": {
        # corridor shifted up
        (0, 2), (0, 3),
        (2, 2), (2, 3),
    },

    "Taps": {
        (1, 2): 10,         # upper tap
        (3, 3): 10,         # lower tap
    },

    "Plants": {
        (0, 0): 5,         # top-left
        (4, 5): 5,         # bottom-right
    },

    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0) : [5,7],
        (4, 5) : [5,7],
    },
    "seed": 45,
    "horizon": 70,
}
problem_new2_version3 = {
    "Size":  (5, 6),

    "Walls": {
        # corridor shifted up
        (0, 2), (0, 3),
        (2, 2), (2, 3),
    },

    "Taps": {
        (1, 2): 10,         # upper tap
        (3, 3): 10,         # lower tap
    },

    "Plants": {
        (0, 0): 5,         # top-left
        (4, 5): 5,         # bottom-right
    },

    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 20,
    "plants_reward": {
        (0, 0) : [5,7,9],
        (4, 5) : [5,7],
    },
    "seed": 45,
    "horizon": 30,
}
problem_new2_version4 = {
    "Size":  (5, 6),

    "Walls": {
        # corridor shifted up
        (0, 2), (0, 3),
        (2, 2), (2, 3),
    },

    "Taps": {
        (1, 2): 10,         # upper tap
        (3, 3): 10,         # lower tap
    },

    "Plants": {
        (0, 0): 5,         # top-left
        (4, 5): 5,         # bottom-right
    },

    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob":{
        10: 0.7,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0) : [5,7],
        (4, 5) : [5,7],
    },
    "seed": 45,
    "horizon": 40,
}


problem_new3_version1 = {
    "Size":  (10, 4),
    "Walls": {
        (0,1),(1, 1), (2, 1), (3, 1), (4, 1), (6, 1),
        (7, 1), (8, 1), (9, 1),(4,2), (4,3),(6,2), (6,3)
    },

    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },

    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,    # upper-right corrido
        (9, 0): 10,   
    },

    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),   # bottom-left area near the tap side
        11: (7, 0, 0, 20),   # bottom-left area near the tap side
    },
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0) : [1,3],
        (9, 0) : [1,3],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new3_version2 = {
    "Size":  (10, 4),
    "Walls": {
        (0,1),(1, 1), (2, 1), (3, 1), (4, 1), (6, 1),
        (7, 1), (8, 1), (9, 1),(4,2), (4,3),(6,2), (6,3)
    },

    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },

    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,    # upper-right corrido
        (9, 0): 10,   
    },

    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),   # bottom-left area near the tap side
        11: (7, 0, 0, 20),   # bottom-left area near the tap side
    },
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.8,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0) : [1,3],
        (9, 0) : [1,3],
    },
    "seed": 45,
    "horizon": 50,
}


problem_new3_version3 = {
    "Size":  (10, 4),
    "Walls": {
        (0,1),(1, 1), (2, 1), (3, 1), (4, 1), (6, 1),
        (7, 1), (8, 1), (9, 1),(4,2), (4,3),(6,2), (6,3)
    },

    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },

    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 5,    # upper-right corrido
        (9, 0): 5,   
    },

    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),   # bottom-left area near the tap side
        11: (7, 0, 0, 20),   # bottom-left area near the tap side
    },
    "robot_chosen_action_prob":{
        10: 0.95,
        11: 0.0001,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0) : [1,3],
        (9, 0) : [1,3],
    },
    "seed": 45,
    "horizon": 70,
}
#reset ? 
problem_new4_version1 = {
    "Size":  (10, 10),

    "Walls": set(),   # completely open grid

    "Taps": {
        (8, 8): 24,      },

    "Plants": {
        (0, 0): 5,    # top-left
        (0, 9): 5,    # top-right
        (9, 0): 5,    # bottom-left
        (9, 9): 5,    # bottom-right
        # total need = 20
    },

    "Robots": {
        10: (8, 9, 0, 5),   
    },
       "robot_chosen_action_prob":{
        10: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0) : [1,3],
        (9, 0) : [1,3],
        (9, 0) : [1,3],
        (9, 9) : [1,3],
    },
    "seed": 45,
    "horizon": 70,
}

#reset ? 
problem_new4_version2 = {
    "Size":  (10, 10),

    "Walls": set(),   # completely open grid

    "Taps": {
        (8, 8): 24,      },

    "Plants": {
        (0, 0): 5,    # top-left
        (0, 9): 5,    # top-right
        (9, 0): 5,    # bottom-left
        (9, 9): 5,    # bottom-right
        # total need = 20
    },

    "Robots": {
        10: (8, 9, 0, 5),   
    },
       "robot_chosen_action_prob":{
        10: 0.85,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0) : [1,3],
        (9, 0) : [1,3],
        (9, 0) : [1,3],
        (9, 9) : [1,3],
    },
    "seed": 45,
    "horizon": 40,
}


def main():
    debug_mode = False
    n_runs = 5
    
    # ANSI Color codes for terminal output
    RED = "\033[91m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    # Baselines extracted from Solution_second.txt
    baselines = {
        "problem_pdf": 21.766667,
        "problem_pdf2": 27.0,
        "problem_pdf3": 40.366667,
        "problem_new1_version1": 62.966667,
        "problem_new1_version2": 87.500000,
        "problem_new1_version3": 26.833333,
        "problem_new2_version1": 26.600000,
        "problem_new2_version2": 86.066667,
        "problem_new2_version3": 46.533333,
        "problem_new2_version4": 39.733333,
        "problem_new3_version1": 2.933333,
        "problem_new3_version2": 4.033333,
        "problem_new3_version3": 5.900000,
        "problem_new4_version1": 38.033333,
        "problem_new4_version2": 16.566667,
    }

    problems = [
        # ("problem_pdf", problem_pdf), # PASS
        # ("problem_pdf2", problem_pdf2), # PASS
        # ("problem_pdf3", problem_pdf3), # PASS
        #("problem_new1_version1", problem_new1_version1),
        #("problem_new1_version2", problem_new1_version2), # PASS
        #("problem_new1_version3", problem_new1_version3), # PASS
        #  ("problem_new2_version1", problem_new2_version1), # PASS
        # ("problem_new2_version2", problem_new2_version2), # PASS
        # ("problem_new2_version3", problem_new2_version3), # PASS
        # ("problem_new2_version4", problem_new2_version4), # PASS
        # ("problem_new3_version1", problem_new3_version1), # PASS
        # ("problem_new3_version2", problem_new3_version2), # PASS
        # ("problem_new3_version3", problem_new3_version3), # PASS
        # ("problem_new4_version1", problem_new4_version1), # PASS
        ("problem_new4_version2", problem_new4_version2), 
    ]

    for prob_name, problem in problems:
        total_reward = 0.0
        total_time = 0
        for seed in range(n_runs):
            problem["seed"] = seed
            game = ext_plant.create_pressure_plate_game((problem, debug_mode))
            start_time = time.time()
            run_reward = solve(game)
            total_reward += run_reward
            duration = time.time() - start_time
            total_time += duration
        
        avg_reward = total_reward / n_runs
        avg_time = total_time / n_runs

        # Comparison Logic
        baseline = baselines.get(prob_name)
        status_msg = ""
        
        if baseline is not None:
            # Using a tiny epsilon for float comparison safety
            if avg_reward >= (baseline - 0.0001):
                status_msg = f" -> {GREEN}PASSED (Baseline: {baseline}){RESET}"
            else:
                status_msg = f" -> {RED}FAILED (Baseline: {baseline}){RESET}"
        
        print(f"\n{CYAN}{prob_name}{RESET}: Average reward over {n_runs} runs: {avg_reward}{status_msg}")
        print(f"{prob_name}: Average time over {n_runs} runs: {avg_time:.4f}s")




if __name__ == "__main__":
    main()
