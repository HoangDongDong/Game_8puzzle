# Complete 8-Puzzle Solver
# To run: Save as a .py file and execute: python your_filename.py
# Make sure you have Pygame installed: pip install pygame

import heapq
import collections
import pygame
import random
import tkinter as tk
from tkinter import simpledialog, messagebox
import math
import time

# --- Tkinter Root Management ---
_tkinter_root_instance = None
def get_tk_root():
    global _tkinter_root_instance
    if _tkinter_root_instance is None or not _tkinter_root_instance.winfo_exists():
        try: _tkinter_root_instance = tk.Tk(); _tkinter_root_instance.withdraw()
        except Exception: _tkinter_root_instance = None
    return _tkinter_root_instance
def destroy_tk_root():
    global _tkinter_root_instance
    if _tkinter_root_instance and _tkinter_root_instance.winfo_exists():
        try: _tkinter_root_instance.destroy()
        except Exception: pass
    _tkinter_root_instance = None

# --- Utility Functions ---
def show_notification(message, title="Thông báo"):
    root = get_tk_root()
    if root:
        try: messagebox.showinfo(title, message, parent=root)
        except Exception as e: print(f"Lỗi hiển thị thông báo Tkinter: {e}. Nội dung: {message}")
    else: print(f"Thông báo (Tkinter không khả dụng) - {title}: {message}")

def get_custom_initial_state_from_user():
    root = get_tk_root()
    if not root: print("Hộp thoại (Tkinter không khả dụng): Không thể lấy trạng thái ban đầu tùy chỉnh."); return None
    input_str = None
    try: input_str = simpledialog.askstring("Nhập Trạng Thái Ban Đầu", "Nhập 9 số (0-8, cách nhau bởi dấu cách hoặc phẩy, 0 là ô trống):", parent=root)
    except Exception as e: show_notification(f"Lỗi khi mở hộp thoại nhập: {e}"); return None
    if input_str is None: return None
    try:
        parts = input_str.replace(",", " ").split(); nums = [int(p) for p in parts]
        if len(parts) != 9 or sorted(nums) != list(range(9)): raise ValueError("Đầu vào không hợp lệ.")
        flat_list_no_zero = [n for n in nums if n != 0]
        inversions = sum(1 for i in range(len(flat_list_no_zero)) for j in range(i + 1, len(flat_list_no_zero)) if flat_list_no_zero[i] > flat_list_no_zero[j])
        if inversions % 2 != 0:
            show_notification(f"Trạng thái không thể giải (số nghịch đảo lẻ: {inversions}). Vui lòng thử lại."); return None
        return tuple(tuple(nums[i:i+3]) for i in range(0, 9, 3))
    except Exception as e: show_notification(f"Đầu vào không hợp lệ: {e}"); return None

# --- Game Logic ---
initial_state_config = ((2, 6, 5), (8, 7, 0), (4, 3, 1))
goal = ((1, 2, 3), (4, 5, 6), (7, 8, 0))
_goal_positions_cache = {}
def _precompute_goal_positions(target_goal=goal):
    global _goal_positions_cache; _goal_positions_cache.clear()
    for r, row in enumerate(target_goal):
        for c, val in enumerate(row):
            if val != 0: _goal_positions_cache[val] = (r, c)
_precompute_goal_positions(goal)
def find_blank(state_tuple):
    for r, row in enumerate(state_tuple):
        for c, val in enumerate(row):
            if val == 0: return r, c
    return -1, -1
def manhattan_distance(state_tuple, target_goal=goal):
    dist = 0
    pos_map = _goal_positions_cache if target_goal == goal else {val:(r,c) for r, row in enumerate(target_goal) for c, val in enumerate(row) if val != 0}
    for r, row_val in enumerate(state_tuple):
        for c, val in enumerate(row_val):
            if val != 0 and val in pos_map: gr, gc = pos_map[val]; dist += abs(r-gr)+abs(c-gc)
    return dist
def generate_moves(state_tuple):
    r_blank, c_blank = find_blank(state_tuple); moves = []
    if r_blank == -1: return moves
    action_map = {'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    for act_name,(dr,dc) in action_map.items():
        nr,nc = r_blank+dr,c_blank+dc
        if 0<=nr<3 and 0<=nc<3:
            nsl=[list(r) for r in state_tuple]; nsl[r_blank][c_blank],nsl[nr][nc]=nsl[nr][nc],nsl[r_blank][c_blank]
            moves.append((tuple(map(tuple,nsl)), act_name))
    return moves
def generate_moves_with_cost(state_tuple): return [(ns, act, 1) for ns, act in generate_moves(state_tuple)]
def perform_action(state_tuple, action_name):
    r_blank,c_blank=find_blank(state_tuple);
    if r_blank==-1: return None
    action_map={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    if action_name not in action_map: return None
    dr,dc=action_map[action_name]; nr,nc=r_blank+dr,c_blank+dc
    if 0<=nr<3 and 0<=nc<3:
        nsl=[list(r)for r in state_tuple]; nsl[r_blank][c_blank],nsl[nr][nc]=nsl[nr][nc],nsl[r_blank][c_blank]
        return tuple(map(tuple,nsl))
    return None

# --- Search Algorithm Implementations (Full, Readable Versions) ---
def bfs(start, goal_state): # Breadth-First Search
    queue = collections.deque([(start, [start])]); visited = {start}
    while queue:
        current_state, path = queue.popleft()
        if current_state == goal_state: return path
        for new_state, _ in generate_moves(current_state):
            if new_state not in visited: visited.add(new_state); queue.append((new_state, path + [new_state]))
    return None

def ucs(start, goal_state): # Uniform Cost Search
    queue = [(0, start, [start])]; visited_costs = {start: 0}
    while queue:
        cost, current_state, current_path = heapq.heappop(queue)
        if cost > visited_costs.get(current_state, float('inf')): continue
        if current_state == goal_state: return current_path
        for new_state, _, step_cost in generate_moves_with_cost(current_state):
            new_cost = cost + step_cost
            if new_cost < visited_costs.get(new_state, float('inf')):
                visited_costs[new_state] = new_cost; heapq.heappush(queue, (new_cost, new_state, current_path + [new_state]))
    return None

def dfs(start, goal_state, max_depth=35): # Depth-First Search
    stack = [(start, [start])]
    while stack:
        current_state, path = stack.pop()
        if current_state == goal_state: return path
        if len(path) > max_depth: continue
        moves = generate_moves(current_state); random.shuffle(moves)
        for new_state, _ in moves:
            if new_state not in path: stack.append((new_state, path + [new_state]))
    return None

def dls_recursive(curr, goal_s, depth_lim, path_curr, visited_path): # Depth-Limited Search Recursive
    if curr == goal_s: return path_curr
    if depth_lim == 0: return None
    visited_path.add(curr)
    for ns, _ in generate_moves(curr):
        if ns not in visited_path:
            res = dls_recursive(ns, goal_s, depth_lim-1, path_curr+[ns], visited_path)
            if res: visited_path.remove(curr); return res
    visited_path.remove(curr); return None

def dls_wrapper(start, goal_s, limit=35): return dls_recursive(start, goal_s, limit, [start], set())

def iddfs(start, goal_s, max_d_limit=30): # Iterative Deepening DFS
    for d in range(max_d_limit+1):
        res = dls_wrapper(start, goal_s, d)
        if res: return res
    return None

def a_star(start, goal_s): # A* Search
    queue = [(manhattan_distance(start,goal_s),0,start,[start])]; open_map={start:0}
    while queue:
        _, g_v, curr, path = heapq.heappop(queue)
        if g_v > open_map.get(curr, float('inf')): continue
        if curr == goal_s: return path
        for ns, _, sc in generate_moves_with_cost(curr):
            ng = g_v + sc
            if ng < open_map.get(ns, float('inf')):
                open_map[ns]=ng; h=manhattan_distance(ns,goal_s); heapq.heappush(queue, (ng+h,ng,ns,path+[ns]))
    return None

def greedy_search(start, goal_s): # Greedy Best-First Search
    queue = [(manhattan_distance(start,goal_s),start,[start])]; visited={start}
    while queue:
        _, curr, path = heapq.heappop(queue)
        if curr == goal_s: return path
        for ns, _, _ in generate_moves_with_cost(curr):
            if ns not in visited:
                visited.add(ns); heapq.heappush(queue, (manhattan_distance(ns,goal_s), ns, path+[ns]))
    return None

def ida_star_search_rec(path_curr, g, threshold, goal_s): # IDA* Recursive Search
    curr = path_curr[-1]; f = g + manhattan_distance(curr, goal_s)
    if f > threshold: return f
    if curr == goal_s: return path_curr
    min_next_thresh = float('inf')
    for ns, _, sc in generate_moves_with_cost(curr):
        if ns not in path_curr:
            res = ida_star_search_rec(path_curr+[ns], g+sc, threshold, goal_s)
            if isinstance(res, list): return res
            min_next_thresh = min(min_next_thresh, res)
    return min_next_thresh

def ida_star(start, goal_s, max_thresh_inc=35): # IDA*
    thresh = manhattan_distance(start, goal_s)
    for _ in range(max_thresh_inc):
        res = ida_star_search_rec([start], 0, thresh, goal_s)
        if isinstance(res, list): return res
        if res == float("inf"): return None
        if res == thresh: return None # Stuck
        thresh = res
        if thresh > 80: return None # Practical limit
    return None

def simple_hill_climbing(start, goal_s, max_steps=1000): # Simple/Steepest Hill Climbing
    curr_s=start; path=[curr_s]; curr_h=manhattan_distance(curr_s,goal_s)
    for _ in range(max_steps):
        if curr_s == goal_s: return path
        neigh=generate_moves(curr_s)
        if not neigh: return path
        best_ns=None; h_best_ns=curr_h
        s_neigh=sorted([(manhattan_distance(ns,goal_s),ns) for ns,d in neigh])
        if s_neigh and s_neigh[0][0] < curr_h: # Strict improvement
            best_opts=[s_v for h,s_v in s_neigh if h==s_neigh[0][0]] # Get all equally best options
            best_ns=random.choice(best_opts) # Choose one randomly
            h_best_ns=s_neigh[0][0]
        if best_ns is None: return path # Stuck
        curr_s=best_ns; curr_h=h_best_ns; path.append(curr_s)
    return path

def stochastic_hill_climbing(start, goal_s, max_iter_climb=200, num_restarts=10):
    best_p=[start]; best_h=manhattan_distance(start,goal_s)
    if best_h==0: return [start]
    for _ in range(num_restarts):
        curr_s=start; curr_p=[curr_s]; curr_h_val=manhattan_distance(curr_s,goal_s)
        for _iter in range(max_iter_climb):
            if curr_s==goal_s: break
            neigh=generate_moves(curr_s)
            if not neigh: break
            bet_neigh=[ns for ns,d in neigh if manhattan_distance(ns,goal_s) < curr_h_val] # Strictly better
            if bet_neigh:
                curr_s=random.choice(bet_neigh); curr_p.append(curr_s); curr_h_val=manhattan_distance(curr_s,goal_s)
            else: break # No strictly better neighbor
        fin_h=manhattan_distance(curr_p[-1],goal_s)
        if fin_h==0: # Found goal
            if best_p[-1]!=goal_s or len(curr_p)<len(best_p): best_p=list(curr_p); best_h=0
        elif best_p[-1]!=goal_s and fin_h<best_h: # Found better non-goal path
            best_p=list(curr_p); best_h=fin_h
    return best_p

def local_beam_search(start, goal_s, beam_w=5, max_iter=200):
    beam=[(manhattan_distance(start,goal_s),[start])]
    for _ in range(max_iter):
        if not beam: return None
        cand_paths={} # {state: (heuristic, path)}
        for h_c, c_p in beam:
            c_s=c_p[-1]
            if c_s==goal_s: return c_p
            for ns,_ in generate_moves(c_s):
                if ns not in c_p: # Avoid cycles in path
                    nh=manhattan_distance(ns,goal_s); np=c_p+[ns]
                    # If new state already a candidate via better or equal path, skip
                    if ns in cand_paths and cand_paths[ns][0]<=nh: continue
                    cand_paths[ns]=(nh,np)
        if not cand_paths: return beam[0][1] if beam else None # No new states
        sort_cand=sorted(list(cand_paths.values()),key=lambda x:x[0]) # Sort by heuristic
        beam=sort_cand[:beam_w] # Select B best
        if beam and beam[0][1][-1]==goal_s: return beam[0][1] # Goal in new beam
    return beam[0][1] if beam else None # Return best after iterations

def simulated_annealing(start, goal_s, init_temp=100.0, cool_r=0.995, min_temp=1e-2, iter_temp=30, max_tot_iter=15000):
    cs=start; ch=manhattan_distance(cs,goal_s); bs=cs; bh=ch; path_log=[cs]; temp=init_temp; total_iter=0
    while temp>min_temp and total_iter<max_tot_iter:
        if cs==goal_s: break
        for _ in range(iter_temp):
            if total_iter>=max_tot_iter: break
            neigh=generate_moves(cs)
            if not neigh: break
            ns,_=random.choice(neigh); nh=manhattan_distance(ns,goal_s); deltah=nh-ch; acc=False
            if deltah<0: acc=True # Always accept better
            elif temp>1e-9: # Avoid division by zero
                try: acc=random.random()<math.exp(-deltah/temp)
                except OverflowError: acc=False # If exp is too large
            if acc:
                cs,ch=ns,nh; path_log.append(cs)
                if ch<bh: bs,bh=cs,ch # Update best state ever seen
            total_iter+=1
            if bs==goal_s: break # If best ever is goal
        if bs==goal_s or total_iter>=max_tot_iter or not neigh: break
        temp*=cool_r
    if bs==goal_s: # If the best state found was the goal
        try: goal_idx=path_log.index(goal_s); return path_log[:goal_idx+1] # Return path up to first time goal was visited
        except ValueError: return a_star(start,goal_s) if start!=goal_s else [start] # Fallback if goal not in log
    return path_log # Return path traversed

def genetic_algorithm_solver(start, goal_s, pop_sz=50, gens=50, mut_r=0.2, cross_r=0.7, max_act_seq=35, tour_sz=3):
    action_choices=['UP','DOWN','LEFT','RIGHT']
    def create_individual(): return [random.choice(action_choices) for _ in range(random.randint(5,max_act_seq))]
    def evaluate_individual(action_sequence, start_s, goal_s):
        current_s=start_s; path=[current_s]; valid_moves=0
        for action in action_sequence:
            next_s=perform_action(current_s,action)
            if next_s is None: break
            current_s=next_s; path.append(current_s); valid_moves+=1
            if current_s==goal_s: break
        h_val=manhattan_distance(current_s,goal_s); fitness=h_val
        if current_s==goal_s: fitness-=10000; fitness+=len(path)*0.01 # Strong reward for goal
        else: fitness+=100 # Penalty for not reaching
        if valid_moves < len(action_sequence): fitness+=50*(len(action_sequence)-valid_moves) # Penalty for invalid sequence
        return path, fitness
    population_actions=[create_individual() for _ in range(pop_sz)]; best_overall_path=None; best_overall_fitness=float('inf')
    for _gen_num in range(gens):
        evaluated_population=[]
        for actions in population_actions:
            path,fitness=evaluate_individual(actions,start,goal_s); evaluated_population.append((fitness,path,actions))
            if fitness<best_overall_fitness: best_overall_fitness,best_overall_path=fitness,path
        evaluated_population.sort(key=lambda x:x[0])
        elites_count=max(1,pop_sz//10); next_population_actions=[ind[2] for ind in evaluated_population[:elites_count]]
        while len(next_population_actions)<pop_sz:
            def tournament_select(pop,k):participants=random.sample(pop,k);participants.sort(key=lambda x:x[0]);return participants[0][2]
            parent1_actions=tournament_select(evaluated_population,tour_sz); parent2_actions=tournament_select(evaluated_population,tour_sz)
            child1_actions,child2_actions=list(parent1_actions),list(parent2_actions) # Make copies for crossover
            if random.random()<cross_r and len(parent1_actions)>1 and len(parent2_actions)>1:
                cx_point1=random.randint(1,len(parent1_actions)-1); cx_point2=random.randint(1,len(parent2_actions)-1)
                child1_actions=parent1_actions[:cx_point1]+parent2_actions[cx_point2:]
                child2_actions=parent2_actions[:cx_point2]+parent1_actions[cx_point1:]
            def mutate(actions_seq_in):
                if not actions_seq_in: return [random.choice(action_choices)]
                actions_seq = list(actions_seq_in)
                for i in range(len(actions_seq)): # Point mutation
                    if random.random()<mut_r: actions_seq[i]=random.choice(action_choices)
                if random.random()<mut_r/2 and len(actions_seq)<max_act_seq: # Add action
                    actions_seq.insert(random.randrange(len(actions_seq)+1),random.choice(action_choices))
                if random.random()<mut_r/2 and len(actions_seq)>1: # Delete action
                    actions_seq.pop(random.randrange(len(actions_seq)))
                return actions_seq
            child1_actions=mutate(child1_actions[:max_act_seq]); child2_actions=mutate(child2_actions[:max_act_seq])
            if child1_actions: next_population_actions.append(child1_actions)
            if len(next_population_actions)<pop_sz and child2_actions: next_population_actions.append(child2_actions)
        population_actions=next_population_actions
        if not population_actions: population_actions=[create_individual() for _ in range(pop_sz)] # Safety repopulate
    return best_overall_path

def backtracking_csp_solver(start, goal_s, max_d=30): return dls_wrapper(start, goal_s, max_d)

def min_conflicts_solver(start, goal_s, max_r=5, max_s_r=500, rand_w_p=0.15):
    best_path=[start]; best_h=manhattan_distance(start,goal_s)
    if best_h==0: return [start]
    MAX_PATH_SAFETY=100
    for _ in range(max_r):
        current_s=start; path_this_restart=[current_s]
        for _step in range(max_s_r):
            if current_s==goal_s:
                if best_path[-1]!=goal_s or len(path_this_restart)<len(best_path): best_path,best_h=list(path_this_restart),0
                break
            current_h_val=manhattan_distance(current_s,goal_s); possible_next_moves=generate_moves(current_s)
            if not possible_next_moves: break
            min_h_candidate=float('inf'); best_candidate_states=[]
            for next_s_tuple,_ in possible_next_moves:
                h_next=manhattan_distance(next_s_tuple,goal_s)
                if h_next<min_h_candidate: min_h_candidate,best_candidate_states=h_next,[next_s_tuple]
                elif h_next==min_h_candidate: best_candidate_states.append(next_s_tuple)
            chosen_next_s=None
            if best_candidate_states and min_h_candidate<current_h_val: chosen_next_s=random.choice(best_candidate_states)
            else:
                if random.random()<rand_w_p and possible_next_moves: chosen_next_s=random.choice(possible_next_moves)[0]
                elif best_candidate_states: chosen_next_s=random.choice(best_candidate_states)
            if chosen_next_s: current_s=chosen_next_s; path_this_restart.append(current_s)
            else: break
            if len(path_this_restart)>MAX_PATH_SAFETY: break
        if current_s==goal_s:
            if best_path[-1]!=goal_s or len(path_this_restart)<len(best_path): best_path,best_h=list(path_this_restart),0
        elif best_path[-1]!=goal_s:
            h_at_restart_end=manhattan_distance(current_s,goal_s)
            if h_at_restart_end<best_h: best_path,best_h=list(path_this_restart),h_at_restart_end
    return best_path

def q_learning_solver(start, goal_s, episodes=150, alpha=0.1, gamma=0.9, epsilon=0.25, max_s_eps=70, max_g_p=70):
    q_table=collections.defaultdict(lambda:collections.defaultdict(float)); best_train_path=None
    for _ep in range(episodes):
        current_s=start; path_episode=[current_s]
        for _step in range(max_s_eps):
            if current_s==goal_s:
                if best_train_path is None or len(path_episode)<len(best_train_path): best_train_path=list(path_episode)
                break
            valid_moves=generate_moves(current_s)
            if not valid_moves: break
            action_name=None
            if random.random()<epsilon: _,action_name=random.choice(valid_moves)
            else:
                max_q_val=-float('inf'); best_actions=[]
                shuffled_moves=list(valid_moves); random.shuffle(shuffled_moves)
                for _,act_cand in shuffled_moves:
                    q_s_a=q_table[current_s][act_cand]
                    if q_s_a>max_q_val: max_q_val,best_actions=q_s_a,[act_cand]
                    elif q_s_a==max_q_val: best_actions.append(act_cand)
                if best_actions: action_name=random.choice(best_actions)
                else: _,action_name=random.choice(valid_moves)
            next_s=perform_action(current_s,action_name)
            if next_s is None: continue
            reward=-1
            if next_s==goal_s: reward=100
            next_s_valid_moves=generate_moves(next_s); max_q_next_state=0.0
            if next_s_valid_moves:
                q_values_for_next_actions=[q_table[next_s][act_prime] for _,act_prime in next_s_valid_moves]
                if q_values_for_next_actions: max_q_next_state=max(q_values_for_next_actions)
            old_q=q_table[current_s][action_name]; q_table[current_s][action_name]=old_q+alpha*(reward+gamma*max_q_next_state-old_q)
            current_s=next_s; path_episode.append(current_s)
    if best_train_path: return best_train_path
    greedy_path=[start]; current_s_greedy=start; visited_in_greedy={start}
    for _ in range(max_g_p):
        if current_s_greedy==goal_s: return greedy_path
        valid_moves_greedy=generate_moves(current_s_greedy)
        if not valid_moves_greedy: break
        best_q_greedy=-float('inf'); action_for_greedy=None; potential_greedy_options=[]
        shuffled_greedy_moves=list(valid_moves_greedy); random.shuffle(shuffled_greedy_moves)
        for next_s_cand,act_str_g in shuffled_greedy_moves:
            if next_s_cand in visited_in_greedy: continue
            q_val=q_table[current_s_greedy][act_str_g]
            if q_val>best_q_greedy: best_q_greedy,potential_greedy_options=q_val,[(act_str_g,next_s_cand)]
            elif q_val==best_q_greedy: potential_greedy_options.append((act_str_g,next_s_cand))
        next_s_chosen_greedy=None
        if potential_greedy_options: action_for_greedy,next_s_chosen_greedy=random.choice(potential_greedy_options)
        else:
            unvisited_options=[(act,ns) for ns,act in shuffled_greedy_moves if ns not in visited_in_greedy]
            if unvisited_options: action_for_greedy,next_s_chosen_greedy=random.choice(unvisited_options)
            else: break
        if action_for_greedy is None or next_s_chosen_greedy is None: break
        current_s_greedy=next_s_chosen_greedy; greedy_path.append(current_s_greedy); visited_in_greedy.add(current_s_greedy)
    return greedy_path if greedy_path and greedy_path[-1]==goal_s else None

def sarsa_solver(start_state, goal_state, episodes=150, alpha=0.1, gamma=0.9, epsilon=0.25, max_steps_per_episode=70, max_greedy_path_len=70):
    q_table=collections.defaultdict(lambda:collections.defaultdict(float)); btp=None
    for _ep in range(episodes):
        current_s=start_state; p_ep=[current_s]
        valid_moves_s=generate_moves(current_s)
        if not valid_moves_s: continue
        action_s_name=None
        if random.random()<epsilon: _,action_s_name=random.choice(valid_moves_s)
        else:
            max_q_s=-float('inf'); best_actions_s=[]
            shuffled_moves_s=list(valid_moves_s);random.shuffle(shuffled_moves_s)
            for _,act_cand_s in shuffled_moves_s:
                q_s_a=q_table[current_s][act_cand_s]
                if q_s_a>max_q_s:max_q_s,best_actions_s=q_s_a,[act_cand_s]
                elif q_s_a==max_q_s:best_actions_s.append(act_cand_s)
            if best_actions_s:action_s_name=random.choice(best_actions_s)
            else:_,action_s_name=random.choice(valid_moves_s)
        for _step in range(max_steps_per_episode):
            if current_s==goal_state:
                if btp is None or len(p_ep)<len(btp):btp=list(p_ep)
                reward=100;old_q=q_table[current_s][action_s_name];q_table[current_s][action_s_name]=old_q+alpha*(reward+gamma*0-old_q)
                break
            next_s=perform_action(current_s,action_s_name)
            if next_s is None:break
            p_ep.append(next_s);reward=-1
            if next_s==goal_state:reward=100
            action_s_prime_name=None;q_s_prime_a_prime=0.0
            if next_s!=goal_state:
                valid_moves_s_prime=generate_moves(next_s)
                if valid_moves_s_prime:
                    if random.random()<epsilon:_,action_s_prime_name=random.choice(valid_moves_s_prime)
                    else:
                        max_q_s_prime=-float('inf');best_actions_s_prime=[]
                        shuffled_moves_s_prime=list(valid_moves_s_prime);random.shuffle(shuffled_moves_s_prime)
                        for _,act_cand_s_prime in shuffled_moves_s_prime:
                            q_val=q_table[next_s][act_cand_s_prime]
                            if q_val>max_q_s_prime:max_q_s_prime,best_actions_s_prime=q_val,[act_cand_s_prime]
                            elif q_val==max_q_s_prime:best_actions_s_prime.append(act_cand_s_prime)
                        if best_actions_s_prime:action_s_prime_name=random.choice(best_actions_s_prime)
                        else:_,action_s_prime_name=random.choice(valid_moves_s_prime)
                    q_s_prime_a_prime=q_table[next_s][action_s_prime_name]
            old_q=q_table[current_s][action_s_name];q_table[current_s][action_s_name]=old_q+alpha*(reward+gamma*q_s_prime_a_prime-old_q)
            current_s=next_s;action_s_name=action_s_prime_name
            if action_s_name is None and current_s!=goal_state:break
    if btp:return btp
    gp=[start_state];cs_g=start_state;vis_g={start_state}
    for _ in range(max_greedy_path_len):
        if cs_g==goal_state:return gp
        mvs_g=generate_moves(cs_g);
        if not mvs_g:break
        bq_g=-float('inf');ag=None;pg_as=[]
        random.shuffle(mvs_g)
        for ns_c,as_g in mvs_g:
            if ns_c in vis_g:continue
            qv=q_table[cs_g][as_g]
            if qv>bq_g:bq_g,pg_as=qv,[(as_g,ns_c)]
            elif qv==bq_g:pg_as.append((as_g,ns_c))
        ns_cg=None
        if pg_as:ag,ns_cg=random.choice(pg_as)
        else:unv=[(ac,nsc)for nsc,ac in mvs_g if nsc not in vis_g];
        if unv:ag,ns_cg=random.choice(unv)
        else:break
        if ag is None or ns_cg is None:break
        cs_g=ns_cg;gp.append(cs_g);vis_g.add(cs_g)
    return gp if gp and gp[-1]==goal_state else None

# --- Stubs for Advanced Algorithms ---
def partial_observation_search_stub(s,g): show_notification("POMDP (Quan sát một phần) liên quan đến trạng thái niềm tin. Không phù hợp với bài toán 8-puzzle có thể quan sát đầy đủ này.", "Thuật toán khái niệm: POMDP"); return None
def no_observation_search_stub(s,g): show_notification("Tìm kiếm không cảm biến (Không quan sát) nhằm tìm một chuỗi hành động phổ quát. Khác với tìm đường đi với trạng thái bắt đầu đã biết.", "Thuật toán khái niệm: Sensorless"); return None
def backtracking_forward_checking_stub(s,g,md=30): show_notification("Quay lui với kiểm tra trước là một kỹ thuật CSP. Sử dụng DLS làm giải pháp thay thế cho tìm đường.", "Lưu ý thuật toán: CSP"); return dls_wrapper(s,g,md)
def dqn_stub(s,g): show_notification("DQN sử dụng Mạng Neural cho xấp xỉ hàm Q trong Học tăng cường. Yêu cầu thư viện ML và thiết lập phức tạp.", "Thuật toán khái niệm: DQN"); return None
def policy_gradient_stub(s,g): show_notification("Phương pháp Policy Gradient học trực tiếp một chính sách (thường là Mạng Neural). Phức tạp, cần thư viện ML.", "Thuật toán khái niệm: Policy Gradient"); return None

# --- Pygame Setup ---
pygame.init(); pygame.font.init()
screen_width,screen_height=1000,990 # Adjusted height for titles and buttons
screen=pygame.display.set_mode((screen_width,screen_height));pygame.display.set_caption("8-Puzzle Solver");clock=pygame.time.Clock()
try:
    font_ui_button=pygame.font.Font(None,26); font_algo_button=pygame.font.Font(None,17); font_info=pygame.font.Font(None,26)
    font_path_step_title=pygame.font.Font(None,18); font_path_matrix_digit=pygame.font.Font(None,14); font_speed_indicator=pygame.font.Font(None,20)
    font_category_title = pygame.font.Font(None, 22)
    font_steps_count = pygame.font.Font(None, 26) # Font for step count
except:
    font_ui_button=pygame.font.SysFont(None,26); font_algo_button=pygame.font.SysFont(None,17); font_info=pygame.font.SysFont(None,26)
    font_path_step_title=pygame.font.SysFont(None,18); font_path_matrix_digit=pygame.font.SysFont(None,14); font_speed_indicator=pygame.font.SysFont(None,20)
    font_category_title = pygame.font.SysFont(None, 22)
    font_steps_count = pygame.font.SysFont(None, 26)
C_WHITE=(255,255,255);C_BLACK=(0,0,0);C_LIGHT_BLUE=(173,216,230);C_DARK_BLUE=(70,130,180);C_LIGHT_GRAY=(211,211,211)
C_TEXT_TILE=C_WHITE;C_TEXT_BTN=C_WHITE;C_BG_BTN=C_BLACK;C_HIGHLIGHT_PATH=(255,100,100);C_BORDER_BTN=C_WHITE
CELL_SIZE=125 # Further reduced grid cells for more UI space
GRID_START_X=250; GRID_START_Y=25 # Adjusted grid start slightly up
GRID_ROWS,GRID_COLS=3,3;PUZZLE_WIDTH=GRID_COLS*CELL_SIZE;PUZZLE_HEIGHT=GRID_ROWS*CELL_SIZE
PATH_MATRIX_CELL_SIZE=13;PATH_MATRIX_WIDTH=3*PATH_MATRIX_CELL_SIZE;PATH_PREVIEW_AREA_WIDTH=PATH_MATRIX_WIDTH+20
PATH_PREVIEW_COLUMN_START_X=screen_width-PATH_PREVIEW_AREA_WIDTH-15;PATH_PREVIEW_COLUMN_START_Y=GRID_START_Y+25
PATH_MATRIX_VERTICAL_SPACING=2;MAX_PATH_STATES_DISPLAY_VERTICAL=max(1,int((PUZZLE_HEIGHT-25)/(PATH_MATRIX_WIDTH+PATH_MATRIX_VERTICAL_SPACING)))

# --- UI Element Drawing Functions ---
def draw_text_in_rect(text, rect, surface, font, tc=C_TEXT_BTN, bg=C_BG_BTN, bc=C_BORDER_BTN, bw=2):
    pygame.draw.rect(surface, bg, rect)
    if bw > 0: pygame.draw.rect(surface, bc, rect, bw)
    ts = font.render(text, True, tc); tr = ts.get_rect(center=rect.center)
    surface.blit(ts, tr)
_mmdfc={};_ptfmg=None;_ptsc={}
def draw_mini_matrix(s, st, sx, sy, hl=False):
    global _mmdfc, font_path_matrix_digit
    if st is None: return
    br = pygame.Rect(sx, sy, PATH_MATRIX_WIDTH, PATH_MATRIX_WIDTH)
    if hl: pygame.draw.rect(s, C_HIGHLIGHT_PATH, br.inflate(4,4), 2)
    for r_idx in range(3):
        for c_idx in range(3):
            v = st[r_idx][c_idx]
            cr = pygame.Rect(sx+c_idx*PATH_MATRIX_CELL_SIZE, sy+r_idx*PATH_MATRIX_CELL_SIZE, PATH_MATRIX_CELL_SIZE, PATH_MATRIX_CELL_SIZE)
            pygame.draw.rect(s, C_DARK_BLUE if v!=0 else C_LIGHT_GRAY, cr)
            pygame.draw.rect(s, C_WHITE, cr, 1)
            if v!=0:
                if v not in _mmdfc: _mmdfc[v] = font_path_matrix_digit.render(str(v),True,C_TEXT_TILE)
                tss = _mmdfc[v]; s.blit(tss, tss.get_rect(center=cr.center))
def draw_puzzle_grid(s, mt, hltv=None, hltp=None):
    global _ptfmg, _ptsc
    if _ptfmg is None:
        tfs = int(CELL_SIZE*0.55)
        try: _ptfmg = pygame.font.Font(None, tfs)
        except pygame.error: _ptfmg = pygame.font.SysFont(None, tfs)
    cdm = [list(r) for r in mt]
    if hltv is not None:
        for r_idx in range(GRID_ROWS):
            try: c_idx = cdm[r_idx].index(hltv); cdm[r_idx][c_idx]=0; break
            except ValueError: pass
    for r_idx in range(GRID_ROWS):
        for c_idx in range(GRID_COLS):
            v=cdm[r_idx][c_idx]; cr=pygame.Rect(GRID_START_X+c_idx*CELL_SIZE,GRID_START_Y+r_idx*CELL_SIZE,CELL_SIZE,CELL_SIZE)
            pygame.draw.rect(s,C_DARK_BLUE if v!=0 else C_LIGHT_GRAY,cr)
            pygame.draw.rect(s,C_WHITE if v!=0 else C_BLACK,cr,3 if v!=0 else 2)
            if v!=0:
                if v not in _ptsc: _ptsc[v]=_ptfmg.render(str(v),True,C_TEXT_TILE)
                ts=_ptsc[v]; s.blit(ts,ts.get_rect(center=cr.center))
    if hltv is not None and hltp is not None:
        atr=pygame.Rect(hltp[0],hltp[1],CELL_SIZE,CELL_SIZE); pygame.draw.rect(s,C_DARK_BLUE,atr); pygame.draw.rect(s,C_WHITE,atr,3)
        if hltv not in _ptsc: _ptsc[hltv]=_ptfmg.render(str(hltv),True,C_TEXT_TILE)
        ts_hl=_ptsc[hltv]; s.blit(ts_hl,ts_hl.get_rect(center=atr.center))

def get_moving_tile_info_for_animation(ps,cs):
    if ps is None or cs is None or ps==cs: return None
    prb,pcb=find_blank(ps); crb,ccb=find_blank(cs)
    if (prb,pcb)==(-1,-1)or(crb,ccb)==(-1,-1): return None
    mtv=cs[prb][pcb];
    if mtv==0: return None
    tsr,tsc=crb,ccb; ter,tec=prb,pcb
    sxy=(GRID_START_X+tsc*CELL_SIZE,GRID_START_Y+tsr*CELL_SIZE); exy=(GRID_START_X+tec*CELL_SIZE,GRID_START_Y+ter*CELL_SIZE)
    return mtv,sxy,exy

# --- Button Definitions ---
UI_BTN_W,UI_BTN_H = 170, 40; UI_S_X,UI_S_Y = 10, 8 # Slightly more compact sidebar
btn_set_initial_rect=pygame.Rect(UI_S_X,GRID_START_Y,UI_BTN_W,UI_BTN_H)
btn_reset_rect=pygame.Rect(UI_S_X,btn_set_initial_rect.bottom+UI_S_Y,UI_BTN_W,UI_BTN_H)
btn_stop_rect=pygame.Rect(UI_S_X,btn_reset_rect.bottom+UI_S_Y,UI_BTN_W,UI_BTN_H)
SPEED_BTN_W=(UI_BTN_W-UI_S_X//2)//2; speed_btns_y=btn_stop_rect.bottom+UI_S_Y+5
btn_speed_up_rect=pygame.Rect(btn_stop_rect.left,speed_btns_y,SPEED_BTN_W,UI_BTN_H-7)
btn_speed_down_rect=pygame.Rect(btn_speed_up_rect.right+UI_S_X//2,speed_btns_y,SPEED_BTN_W,UI_BTN_H-7)

ALGO_BTN_W,ALGO_BTN_H=135, 28; ALGO_S_X,ALGO_S_Y=6, 5
ALGOS_PER_ROW=4; CATEGORY_TITLE_HEIGHT=28; CATEGORY_SPACING=8

algo_categories = {
    "Uninformed Search": [("BFS",bfs),("DFS",dfs),("IDDFS",iddfs),("UCS",ucs)],
    "Informed Search": [("Greedy",greedy_search),("A*",a_star),("IDA*",ida_star)],
    "Local Search": [
        ("Simple HC",simple_hill_climbing), ("Steepest HC", simple_hill_climbing),
        ("Sto. HC",stochastic_hill_climbing), ("Local Beam",local_beam_search),
        ("SimAnneal",simulated_annealing), ("Genetic", genetic_algorithm_solver)
    ],
    "CSPs": [
        ("Backtrack",backtracking_csp_solver), ("BacktrackFC",backtracking_forward_checking_stub),
        ("MinConflict",min_conflicts_solver)
    ],
    "Reinforcement Learning": [
        ("Q-Learn",q_learning_solver), ("SARSA",sarsa_solver)
    ],
    "Complex Environments (Stubs)": [
        ("POMDP Stb",partial_observation_search_stub), ("NoObs Stb",no_observation_search_stub)
    ]
}
all_algo_buttons_ui_elements = []
current_algo_y_offset = GRID_START_Y + PUZZLE_HEIGHT + 10 # Start Y for first category title

for category_name, algo_list in algo_categories.items():
    title_surf = font_category_title.render(category_name + ":", True, C_BLACK)
    algo_btn_area_total_width = ALGOS_PER_ROW * ALGO_BTN_W + (ALGOS_PER_ROW - 1) * ALGO_S_X
    title_x_pos = GRID_START_X + (PUZZLE_WIDTH - algo_btn_area_total_width) / 2 + (algo_btn_area_total_width - title_surf.get_width()) / 2
    if title_x_pos < GRID_START_X : title_x_pos = GRID_START_X
    title_rect = title_surf.get_rect(left=int(title_x_pos), top=int(current_algo_y_offset))
    all_algo_buttons_ui_elements.append({'type':'title', 'surface':title_surf, 'rect':title_rect})
    current_algo_y_offset += CATEGORY_TITLE_HEIGHT

    algo_btn_x_start_for_category = GRID_START_X + (PUZZLE_WIDTH - algo_btn_area_total_width) / 2
    if algo_btn_x_start_for_category < UI_S_X : algo_btn_x_start_for_category = GRID_START_X

    for i,(text,func) in enumerate(algo_list):
        row,col = i//ALGOS_PER_ROW, i%ALGOS_PER_ROW
        rect = pygame.Rect(int(algo_btn_x_start_for_category + col*(ALGO_BTN_W+ALGO_S_X)),
                           int(current_algo_y_offset + row*(ALGO_BTN_H+ALGO_S_Y)),
                           ALGO_BTN_W,ALGO_BTN_H)
        all_algo_buttons_ui_elements.append({'type':'button','rect':rect,'text':text,'func':func,'name':text})
    num_rows_cat = (len(algo_list) + ALGOS_PER_ROW - 1) // ALGOS_PER_ROW
    current_algo_y_offset += num_rows_cat * (ALGO_BTN_H + ALGO_S_Y) + CATEGORY_SPACING

# --- Game Loop Variables ---
running=True; cur_puz_disp_state=tuple(map(tuple,initial_state_config)); act_sol_path=None
cur_step_idx_path=0; is_sol_anim_act=False; notif_shown_solve=False; last_algo_exec_time=0.0
is_tile_anim=False; anim_tile_val=0; anim_cur_xy=[0.,0.]; anim_s_xy=[0.,0.]; anim_t_xy=[0.,0.]
anim_tot_frames=12; anim_cur_frame=0; anim_base_state_draw=None
MIN_ANIM_FRAMES=3; MAX_ANIM_FRAMES=45
solution_steps = 0 # Variable to store number of steps

# --- Main Game Loop ---
try:
    while running:
        screen.fill(C_LIGHT_BLUE); mouse_pos=pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:running=False
            elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                if is_tile_anim:continue
                if btn_set_initial_rect.collidepoint(mouse_pos):
                    ni=get_custom_initial_state_from_user()
                    if ni:initial_state_config=ni;_precompute_goal_positions(goal);cur_puz_disp_state=tuple(map(tuple,initial_state_config));act_sol_path=None;is_sol_anim_act=False;notif_shown_solve=False;last_algo_exec_time=0.0;solution_steps=0;show_notification("Trạng thái ban đầu đã được cập nhật.")
                elif btn_reset_rect.collidepoint(mouse_pos):cur_puz_disp_state=tuple(map(tuple,initial_state_config));act_sol_path=None;is_sol_anim_act=False;notif_shown_solve=False;last_algo_exec_time=0.0;solution_steps=0
                elif btn_stop_rect.collidepoint(mouse_pos):
                    if is_sol_anim_act:is_sol_anim_act=False;is_tile_anim=False;show_notification("Animation đã dừng.");notif_shown_solve=True
                    if act_sol_path and cur_step_idx_path<len(act_sol_path):cur_puz_disp_state=act_sol_path[cur_step_idx_path]
                elif btn_speed_up_rect.collidepoint(mouse_pos):anim_tot_frames=max(MIN_ANIM_FRAMES,anim_tot_frames-3)
                elif btn_speed_down_rect.collidepoint(mouse_pos):anim_tot_frames=min(MAX_ANIM_FRAMES,anim_tot_frames+3)
                else:
                    for ui_el in all_algo_buttons_ui_elements:
                        if ui_el['type'] == 'button' and ui_el['rect'].collidepoint(mouse_pos):
                            bd = ui_el; an,af=bd['name'],bd['func'];ss_s=tuple(map(tuple,cur_puz_disp_state));last_algo_exec_time=0.0;notif_shown_solve=False;act_sol_path=None;is_sol_anim_act=False;is_tile_anim=False;solution_steps=0
                            print(f"Đang chạy thuật toán {an}...");ts=time.perf_counter();tp=af(ss_s,goal);last_algo_exec_time=time.perf_counter()-ts;print(f"{an} mất {last_algo_exec_time:.4f} giây để hoàn thành.")
                            if tp and isinstance(tp,list)and len(tp)>0:
                                act_sol_path=tp;cur_puz_disp_state=act_sol_path[0];cur_step_idx_path=0; solution_steps = len(act_sol_path) -1
                                if act_sol_path[-1]==goal:is_sol_anim_act=True;print(f"{an} tìm thấy giải pháp với {solution_steps} bước.")
                                else:is_sol_anim_act=False;cur_puz_disp_state=act_sol_path[-1];
                                if not is_sol_anim_act and not notif_shown_solve:show_notification(f"{an}: Thuật toán kết thúc. Trạng thái cuối được hiển thị (không phải đích). Số bước: {solution_steps}");notif_shown_solve=True;print(f"{an} không đạt đích. Độ dài đường đi: {solution_steps}. H heuristic cuối = {manhattan_distance(act_sol_path[-1])}")
                            else:act_sol_path=None; solution_steps = 0
                            if not act_sol_path and not notif_shown_solve:
                                if "Stb" not in an and "Note" not in an : show_notification(f"{an}: Không tìm thấy giải pháp hoặc đường đi không hợp lệ.");
                                notif_shown_solve=True; print(f"{an}: Không có giải pháp.")
                            break
        if not is_tile_anim and is_sol_anim_act and act_sol_path:
            if cur_step_idx_path<len(act_sol_path)-1:
                sb,sa=act_sol_path[cur_step_idx_path],act_sol_path[cur_step_idx_path+1];mi=get_moving_tile_info_for_animation(sb,sa)
                if mi:anim_tile_val,anim_s_xy_tuple,anim_t_xy_tuple=mi;anim_s_xy[:]=anim_s_xy_tuple; anim_t_xy[:]=anim_t_xy_tuple; anim_cur_xy[:]=anim_s_xy;anim_base_state_draw=sb;is_tile_anim=True;anim_cur_frame=0
                else:is_sol_anim_act=False;print("Lỗi: Thông tin di chuyển không hợp lệ. Stop animation.")
            else:is_sol_anim_act=False
            if not is_sol_anim_act and act_sol_path:cur_puz_disp_state=act_sol_path[-1]
            if cur_puz_disp_state==goal and not notif_shown_solve and not is_sol_anim_act:show_notification(f"Đã đạt đến trạng thái đích! Số bước: {solution_steps}");notif_shown_solve=True
            elif cur_puz_disp_state!=goal and act_sol_path and not notif_shown_solve and not is_sol_anim_act:show_notification(f"Đường đi đã kết thúc. Trạng thái cuối được hiển thị. Số bước: {solution_steps}");notif_shown_solve=True
        if is_tile_anim:
            anim_cur_frame+=1;ratio=min(1.,anim_cur_frame/anim_tot_frames);anim_cur_xy[0]=anim_s_xy[0]+(anim_t_xy[0]-anim_s_xy[0])*ratio;anim_cur_xy[1]=anim_s_xy[1]+(anim_t_xy[1]-anim_s_xy[1])*ratio
            draw_puzzle_grid(screen,anim_base_state_draw,anim_tile_val,anim_cur_xy)
            if anim_cur_frame>=anim_tot_frames:is_tile_anim=False;cur_step_idx_path+=1
            if not is_tile_anim and act_sol_path and cur_step_idx_path<len(act_sol_path):cur_puz_disp_state=act_sol_path[cur_step_idx_path]
        else:draw_puzzle_grid(screen,cur_puz_disp_state)
        draw_text_in_rect("Set Start",btn_set_initial_rect,screen,font_ui_button);draw_text_in_rect("Reset",btn_reset_rect,screen,font_ui_button);draw_text_in_rect("Stop Anim",btn_stop_rect,screen,font_ui_button)
        draw_text_in_rect("Speed+",btn_speed_up_rect,screen,font_ui_button);draw_text_in_rect("Speed-",btn_speed_down_rect,screen,font_ui_button)
        for ui_el in all_algo_buttons_ui_elements:
            if ui_el['type']=='button':draw_text_in_rect(ui_el['text'],ui_el['rect'],screen,font_algo_button)
            elif ui_el['type']=='title':screen.blit(ui_el['surface'],ui_el['rect'])
        
        # Info text (Speed, Time, Steps)
        info_text_y_start = btn_speed_up_rect.bottom + 10
        stxt="Speed Anim: ";rspd=(anim_tot_frames-MIN_ANIM_FRAMES)/(MAX_ANIM_FRAMES-MIN_ANIM_FRAMES) if MAX_ANIM_FRAMES>MIN_ANIM_FRAMES else 0.5
        if rspd<=0.15:stxt+="Nhanh Nhat"
        elif rspd<=0.4:stxt+="Nhanh"
        elif rspd<=0.75:stxt+="Trung Binh"
        else:stxt+="Cham"
        ss=font_speed_indicator.render(stxt,True,C_BLACK);sr=ss.get_rect(left=btn_speed_up_rect.left,top=info_text_y_start);screen.blit(ss,sr)
        info_text_y_start = sr.bottom + 5

        rtval=f"{last_algo_exec_time:.4f}s"if last_algo_exec_time>0 else"N/A";rts=font_info.render(f"Time: {rtval}",True,C_BLACK)
        rts_rect = rts.get_rect(left=btn_speed_up_rect.left, top=info_text_y_start); screen.blit(rts,rts_rect)
        info_text_y_start = rts_rect.bottom + 5

        steps_text_val = f"{solution_steps}" if solution_steps > 0 or (act_sol_path and len(act_sol_path) > 0) else "N/A"
        steps_surf = font_steps_count.render(f"Step: {steps_text_val}", True, C_BLACK)
        steps_rect = steps_surf.get_rect(left=btn_speed_up_rect.left, top=info_text_y_start); screen.blit(steps_surf, steps_rect)

        if act_sol_path:
            ptt="Các Bước Tiếp Theo:"if is_sol_anim_act else"Đường Đi (Dừng/Đã giải):";ts=font_path_step_title.render(ptt,True,C_BLACK);screen.blit(ts,(PATH_PREVIEW_COLUMN_START_X,PATH_PREVIEW_COLUMN_START_Y-25))
            pi=cur_step_idx_path
            if not is_sol_anim_act:
                try:pi=act_sol_path.index(cur_puz_disp_state)
                except ValueError:pi=0
            dy=PATH_PREVIEW_COLUMN_START_Y
            for i in range(MAX_PATH_STATES_DISPLAY_VERTICAL):
                api=pi+i
                if api<len(act_sol_path):
                    sshow=act_sol_path[api];ihl=False
                    if is_sol_anim_act:
                        if is_tile_anim and api == cur_step_idx_path + 1: ihl = True
                        elif not is_tile_anim and api == cur_step_idx_path: ihl = True
                    elif sshow == cur_puz_disp_state and api == pi: ihl = True
                    draw_mini_matrix(screen,sshow,PATH_PREVIEW_COLUMN_START_X,dy,ihl);dy+=PATH_MATRIX_WIDTH+PATH_MATRIX_VERTICAL_SPACING
                    if dy+PATH_MATRIX_WIDTH>screen_height-10:break
        pygame.display.flip();clock.tick(60)
except KeyboardInterrupt:print("Chương trình bị dừng bởi người dùng.")
except Exception as e:print(f"Lỗi không mong muốn xảy ra: {e}");import traceback;traceback.print_exc()
finally:destroy_tk_root();pygame.quit()