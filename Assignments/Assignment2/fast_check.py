import ex2
import ext_plant
import ex2_check
import sys

def run_check():
    # Redirect stdout
    original_stdout = sys.stdout
    with open("trace.log", "w", encoding='utf-8') as log_file:
        sys.stdout = log_file
        
        try:
            prob = ex2_check.problem_new1_version1
            print("Running check for problem_new1_version1...")
            
            total_reward = 0
            runs = 1 # Debug single run
            for i in range(runs):
                prob["seed"] = 123 # Deterministic debug
                game = ext_plant.create_pressure_plate_game((prob, False))
                rew = ex2_check.solve(game)
                game.show_history()
                total_reward += rew
                print(f"Run {i}: Reward {rew}")
                
            avg = total_reward / runs
            print(f"Average Reward: {avg}")
        finally:
            sys.stdout = original_stdout
            
    # Read back and print summary to console
    with open("trace.log", "r", encoding='utf-8') as f:
        content = f.read()
        if "Reward" in content:
            for line in content.splitlines():
                if "Reward" in line:
                     print(line)

if __name__ == "__main__":
    run_check()
