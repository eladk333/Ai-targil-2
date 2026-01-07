import tkinter as tk
from tkinter import ttk, messagebox
import ext_plant
import ex2  # Your solution file
import time

# ==========================================
#        PRESET PROBLEMS DATA
# ==========================================
PRESET_PROBLEMS = {
    "problem_pdf": {
        "Size":   (3, 3),
        "Walls":  {(0, 1), (2, 1)},
        "Taps":   {(1, 1): 6},
        "Plants": {(2, 0): 2, (0, 2): 3},
        "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
        "robot_chosen_action_prob":{10: 0.95, 11: 0.9},
        "goal_reward": 10,
        "plants_reward": {(0, 2): [1,2,3,4], (2, 0): [1,2,3,4]},
        "seed": 45,
        "horizon": 30,
    },
    "problem_pdf2": {
        "Size":   (3, 3),
        "Walls":  {(0, 1), (2, 1)},
        "Taps":   {(1, 1): 6},
        "Plants": {(2, 0): 2, (0, 2): 3},
        "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
        "robot_chosen_action_prob":{10: 0.9, 11: 0.8},
        "goal_reward": 12,
        "plants_reward": {(0, 2): [1,3,5,7], (2, 0): [1,2,3,4]},
        "seed": 45,
        "horizon": 35,
    },
    "problem_pdf3": {
        "Size":   (3, 3),
        "Walls":  {(0, 1), (2, 1)},
        "Taps":   {(1, 1): 6},
        "Plants": {(2, 0): 2, (0, 2): 3},
        "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
        "robot_chosen_action_prob":{10: 0.7, 11: 0.6},
        "goal_reward": 30,
        "plants_reward": {(0, 2): [1,2,3,4], (2, 0): [10,11,12,13]},
        "seed": 45,
        "horizon": 30,
    },
    "problem_new1_version1": {
        "Size":  (5, 6),
        "Walls": {(1, 2), (1, 3), (3, 2), (3, 3)},
        "Taps": {(2, 2): 12},
        "Plants": {(0, 1): 3, (4, 5): 6},
        "Robots": {10: (2, 1, 0, 6), 11: (2, 4, 0, 3)},
        "robot_chosen_action_prob":{10: 0.9, 11: 0.95},
        "goal_reward": 30,
        "plants_reward": {(4, 5): [1,2,3,4], (0, 1): [10,11,12,13]},
        "seed": 45,
        "horizon": 30,
    },
    "problem_new1_version2": {
        "Size":  (5, 6),
        "Walls": {(1, 2), (1, 3), (3, 2), (3, 3)},
        "Taps": {(2, 2): 12},
        "Plants": {(0, 1): 3, (4, 5): 6},
        "Robots": {10: (2, 1, 0, 6), 11: (2, 4, 0, 3)},
        "robot_chosen_action_prob":{10: 0.6, 11: 0.95},
        "goal_reward": 30,
        "plants_reward": {(4, 5): [1,2,3,4], (0, 1): [10,11,12,13]},
        "seed": 45,
        "horizon": 70,
    },
    "problem_new1_version3": {
        "Size":  (5, 6),
        "Walls": {(1, 2), (1, 3), (3, 2), (3, 3)},
        "Taps": {(2, 2): 12},
        "Plants": {(0, 1): 2, (4, 5): 6},
        "Robots": {10: (2, 1, 0, 6), 11: (2, 4, 0, 3)},
        "robot_chosen_action_prob":{10: 0.6, 11: 0.95},
        "goal_reward": 30,
        "plants_reward": {(4, 5): [1,2,3,4], (0, 1): [10,11,12,13]},
        "seed": 45,
        "horizon": 30,
    },
    "problem_new2_version1": {
        "Size":  (5, 6),
        "Walls": {(0, 2), (0, 3), (2, 2), (2, 3)},
        "Taps": {(1, 2): 10, (3, 3): 10},
        "Plants": {(0, 0): 5, (4, 5): 5},
        "Robots": {10: (1, 1, 0, 5), 11: (3, 4, 0, 4)},
        "robot_chosen_action_prob":{10: 0.95, 11: 0.95},
        "goal_reward": 18,
        "plants_reward": {(0, 0): [5,7], (4, 5): [5,7]},
        "seed": 45,
        "horizon": 30,
    },
    "problem_new2_version2": {
        "Size":  (5, 6),
        "Walls": {(0, 2), (0, 3), (2, 2), (2, 3)},
        "Taps": {(1, 2): 10, (3, 3): 10},
        "Plants": {(0, 0): 5, (4, 5): 5},
        "Robots": {10: (1, 1, 0, 5), 11: (3, 4, 0, 4)},
        "robot_chosen_action_prob":{10: 0.95, 11: 0.95},
        "goal_reward": 18,
        "plants_reward": {(0, 0): [5,7], (4, 5): [5,7]},
        "seed": 45,
        "horizon": 70,
    },
    "problem_new2_version3": {
        "Size":  (5, 6),
        "Walls": {(0, 2), (0, 3), (2, 2), (2, 3)},
        "Taps": {(1, 2): 10, (3, 3): 10},
        "Plants": {(0, 0): 5, (4, 5): 5},
        "Robots": {10: (1, 1, 0, 5), 11: (3, 4, 0, 4)},
        "robot_chosen_action_prob":{10: 0.95, 11: 0.95},
        "goal_reward": 20,
        "plants_reward": {(0, 0): [5,7,9], (4, 5): [5,7]},
        "seed": 45,
        "horizon": 30,
    },
    "problem_new2_version4": {
        "Size":  (5, 6),
        "Walls": {(0, 2), (0, 3), (2, 2), (2, 3)},
        "Taps": {(1, 2): 10, (3, 3): 10},
        "Plants": {(0, 0): 5, (4, 5): 5},
        "Robots": {10: (1, 1, 0, 5), 11: (3, 4, 0, 4)},
        "robot_chosen_action_prob":{10: 0.7, 11: 0.95},
        "goal_reward": 18,
        "plants_reward": {(0, 0): [5,7], (4, 5): [5,7]},
        "seed": 45,
        "horizon": 40,
    },
    "problem_new3_version1": {
        "Size":  (10, 4),
        "Walls": {(0,1),(1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1),(4,2), (4,3),(6,2), (6,3)},
        "Taps": {(5, 3): 20},
        "Plants": {(0, 0): 10, (9, 0): 10},
        "Robots": {10: (2, 0, 0, 2), 11: (7, 0, 0, 20)},
        "robot_chosen_action_prob":{10: 0.95, 11: 0.95},
        "goal_reward": 9,
        "plants_reward": {(0, 0): [1,3], (9, 0): [1,3]},
        "seed": 45,
        "horizon": 30,
    },
    "problem_new3_version2": {
        "Size":  (10, 4),
        "Walls": {(0,1),(1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1),(4,2), (4,3),(6,2), (6,3)},
        "Taps": {(5, 3): 20},
        "Plants": {(0, 0): 10, (9, 0): 10},
        "Robots": {10: (2, 0, 0, 2), 11: (7, 0, 0, 20)},
        "robot_chosen_action_prob":{10: 0.95, 11: 0.8},
        "goal_reward": 9,
        "plants_reward": {(0, 0): [1,3], (9, 0): [1,3]},
        "seed": 45,
        "horizon": 50,
    },
    "problem_new3_version3": {
        "Size":  (10, 4),
        "Walls": {(0,1),(1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1),(4,2), (4,3),(6,2), (6,3)},
        "Taps": {(5, 3): 20},
        "Plants": {(0, 0): 5, (9, 0): 5},
        "Robots": {10: (2, 0, 0, 2), 11: (7, 0, 0, 20)},
        "robot_chosen_action_prob":{10: 0.95, 11: 0.0001},
        "goal_reward": 9,
        "plants_reward": {(0, 0): [1,3], (9, 0): [1,3]},
        "seed": 45,
        "horizon": 70,
    },
    "problem_new4_version1": {
        "Size":  (10, 10),
        "Walls": set(),
        "Taps": {(8, 8): 24},
        "Plants": {(0, 0): 5, (0, 9): 5, (9, 0): 5, (9, 9): 5},
        "Robots": {10: (8, 9, 0, 5)},
        "robot_chosen_action_prob":{10: 0.95},
        "goal_reward": 9,
        "plants_reward": {(0, 0): [1,3], (9, 0): [1,3], (9, 9): [1,3], (0, 9): [1,3]},
        "seed": 45,
        "horizon": 70,
    },
    "problem_new4_version2": {
        "Size":  (10, 10),
        "Walls": set(),
        "Taps": {(8, 8): 24},
        "Plants": {(0, 0): 5, (0, 9): 5, (9, 0): 5, (9, 9): 5},
        "Robots": {10: (8, 9, 0, 5)},
        "robot_chosen_action_prob":{10: 0.85},
        "goal_reward": 9,
        "plants_reward": {(0, 0): [1,3], (9, 0): [1,3], (9, 9): [1,3], (0, 9): [1,3]},
        "seed": 45,
        "horizon": 40,
    }
}

# ==========================================
#         GUI & SIMULATION LOGIC
# ==========================================

class VisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Assignment 2 - Game Replay & Visualizer")
        
        # --- Frame: Selection ---
        self.frame_select = tk.Frame(root, padx=10, pady=10, bg="#f0f0f0")
        self.frame_select.pack(fill="x")

        tk.Label(self.frame_select, text="Select Problem:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.prob_var = tk.StringVar()
        self.combo = ttk.Combobox(self.frame_select, textvariable=self.prob_var, state="readonly", width=30)
        self.combo['values'] = list(PRESET_PROBLEMS.keys())
        
        # DEFAULT SELECTION: problem_new4_version2
        if "problem_new4_version2" in self.combo['values']:
            self.combo.set("problem_new4_version2")
        elif self.combo['values']:
            self.combo.current(0)
            
        self.combo.pack(side=tk.LEFT, padx=5)

        tk.Button(self.frame_select, text="RUN SIMULATION", bg="#ddffdd", command=self.run_simulation).pack(side=tk.LEFT, padx=10)
        
        # --- Frame: Canvas ---
        self.canvas = tk.Canvas(root, width=600, height=500, bg="white")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Frame: Controls ---
        self.frame_controls = tk.Frame(root, padx=10, pady=10)
        self.frame_controls.pack(fill="x")

        # Step Slider
        self.slider_var = tk.IntVar()
        self.slider = tk.Scale(self.frame_controls, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.slider_var, command=self.on_slider_move, label="Step Scrubber")
        self.slider.pack(fill="x", padx=5)

        # Navigation Buttons
        btn_frame = tk.Frame(self.frame_controls)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="<< Prev", command=self.prev_step).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Next >>", command=self.next_step).pack(side=tk.LEFT, padx=5)

        # Info Label
        self.lbl_info = tk.Label(self.frame_controls, text="Run a simulation to see details.", font=("Consolas", 11), fg="#333")
        self.lbl_info.pack(pady=5)

        # Internal Data
        self.history = []
        self.current_problem = {}
        self.walls = set()
        self.rows = 0
        self.cols = 0
        self.cell_size = 50

    def run_simulation(self):
        prob_name = self.prob_var.get()
        if not prob_name: return

        problem = PRESET_PROBLEMS[prob_name].copy()
        
        # Setup Game
        game = ext_plant.create_pressure_plate_game((problem, False))
        
        # Setup Controller
        try:
            policy = ex2.Controller(game)
        except Exception as e:
            messagebox.showerror("Controller Error", f"Error initializing Controller:\n{e}")
            return

        # --- RECORDING LOOP ---
        self.history = []
        
        # Save initial state
        initial_state = game.get_current_state()
        self.history.append({
            'state': initial_state,
            'action': "START",
            'step': 0,
            'reward': 0,
            'total_need': initial_state[3]
        })

        max_steps = game.get_max_steps()
        
        for i in range(max_steps):
            try:
                # Get Action
                state = game.get_current_state()
                action = policy.choose_next_action(state)
                
                # Execute
                game.submit_next_action(action)
                
                # Save Result
                new_state = game.get_current_state()
                self.history.append({
                    'state': new_state,
                    'action': action,
                    'step': i + 1,
                    'reward': game.get_current_reward(),
                    'total_need': new_state[3]
                })

                if game.get_done():
                    break
            except Exception as e:
                messagebox.showerror("Runtime Error", f"Error during step {i}:\n{e}")
                return

        # Setup Visuals
        self.current_problem = problem
        self.walls = set(problem.get("Walls", []))
        self.rows, self.cols = problem["Size"]
        
        # Configure Slider
        self.slider.config(to=len(self.history)-1)
        self.slider_var.set(0)
        
        # Draw Initial
        self.draw_step(0)

    def on_slider_move(self, val):
        idx = int(val)
        self.draw_step(idx)

    def prev_step(self):
        curr = self.slider_var.get()
        if curr > 0:
            self.slider_var.set(curr - 1)
            self.draw_step(curr - 1)

    def next_step(self):
        curr = self.slider_var.get()
        if curr < len(self.history) - 1:
            self.slider_var.set(curr + 1)
            self.draw_step(curr + 1)

    def draw_step(self, index):
        if not self.history: return
        
        data = self.history[index]
        state = data['state']
        # unpack state: (robots_t, plants_t, taps_t, total_water_need)
        robots_t, plants_t, taps_t, _ = state
        
        # Info Update
        info_txt = f"Step: {data['step']} | Action: {data['action']} | Total Need: {data['total_need']} | Reward: {data['reward']:.2f}"
        self.lbl_info.config(text=info_txt)

        # Drawing Logic
        self.canvas.delete("all")
        
        # Auto-scale cell size
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        if c_w < 50: c_w = 600
        if c_h < 50: c_h = 500
        
        self.cell_size = min(c_w // self.cols, c_h // self.rows)
        # Cap max size
        if self.cell_size > 80: self.cell_size = 80

        # Maps for easy drawing
        tap_map = {pos: amt for pos, amt in taps_t}
        plant_map = {pos: need for pos, need in plants_t}
        robot_map = {pos: (rid, load) for rid, pos, load in robots_t}

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                pos = (r, c)
                
                # Base Color
                fill_color = "white"
                
                if pos in self.walls:
                    fill_color = "#444" # Wall Gray
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#ccc")
                
                # Draw Objects
                
                # 1. Taps (Blue)
                if pos in tap_map:
                    self.canvas.create_rectangle(x1+5, y1+5, x2-5, y2-5, fill="#ccf", outline="blue", width=2)
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2 - 10, text="T", font=("Arial", 10, "bold"), fill="blue")
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2 + 10, text=str(tap_map[pos]), font=("Arial", 8), fill="black")

                # 2. Plants (Green)
                if pos in plant_map:
                    need = plant_map[pos]
                    # Fetch static rewards
                    rewards = self.current_problem.get("plants_reward", {}).get(pos, [])
                    
                    self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill="#bfb", outline="green", width=2)
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2 - 12, text="P", font=("Arial", 10, "bold"), fill="green")
                    
                    # Display "Need | Rewards"
                    # Using small font to fit list
                    sub_txt = f"{need} | {rewards}"
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2 + 8, text=sub_txt, font=("Arial", 7), fill="black")

                # 3. Robots (Red)
                if pos in robot_map:
                    rid, load = robot_map[pos]
                    # Get static capacity
                    # Structure in preset is {rid: (r, c, load, capacity)}
                    # We only need capacity (index 3)
                    cap = "?"
                    if rid in self.current_problem.get("Robots", {}):
                         cap = self.current_problem["Robots"][rid][3]

                    self.canvas.create_oval(x1+10, y1+10, x2-10, y2-10, fill="#fbb", outline="red", width=2)
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2 - 5, text=f"R{rid}", font=("Arial", 9, "bold"), fill="#800")
                    
                    # Display "W:Load/C:Cap"
                    sub_txt = f"W:{load}/C:{cap}"
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2 + 10, text=sub_txt, font=("Arial", 7), fill="black")

                # Coords
                self.canvas.create_text(x1+8, y1+8, text=f"{r},{c}", fill="#ddd", font=("Arial", 6))

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizerApp(root)
    root.mainloop()