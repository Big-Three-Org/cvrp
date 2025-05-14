#!/usr/bin/env python3
"""
F-CVRP Solver: Initial solution via Savings, improved by Iterated Local Search
Usage: python3 fcvrp_solver.py <instance_file> <output_solution_file>
"""
import sys
import random
import time
from Parser import load_model

# ---------------------- Utility Functions ----------------------

def compute_distance_matrix(model):
    # Use the provided cost_matrix directly
    return model.cost_matrix

# Route is a list of customer IDs (excluding depot)
# Solution is a list of routes

def route_cost(route, dist):
    if not route:
        return 0
    cost = dist[0][route[0]]  # depot to first
    for i in range(len(route)-1):
        cost += dist[route[i]][route[i+1]]
    cost += dist[route[-1]][0]  # last back to depot
    return cost


def solution_cost(sol, dist):
    return sum(route_cost(r, dist) for r in sol)

# ---------------------- Initial Selection ----------------------

def select_initial_customers(model):
    # For each family, choose the required_visits closest to depot
    selected = []
    for fam in model.families:
        # sort family's nodes by cost from depot
        nodes = fam.nodes
        nodes_sorted = sorted(nodes, key=lambda n: model.cost_matrix[0][n.id])
        sel = [n.id for n in nodes_sorted[:fam.required_visits]]
        selected.extend(sel)
    return set(selected)

# ---------------------- Clarke-Wright Savings ----------------------

def construct_initial_routes(model, dist, selected):
    # Each selected customer starts in its own route
    routes = [[cid] for cid in selected]
    route_load = {i: model.nodes[cid].demand for i, cid in enumerate(selected)}
    # Precompute savings
    savings = []  # (saving, i, j)
    for i in range(len(routes)):
        for j in range(i+1, len(routes)):
            ci = routes[i][0]
            cj = routes[j][0]
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])
    # Merge routes in order of savings if capacity allows
    used = [True]*len(routes)
    for s, i, j in savings:
        if not (used[i] and used[j]):
            continue
        ri, rj = routes[i], routes[j]
        load_i = sum(model.nodes[c].demand for c in ri)
        load_j = sum(model.nodes[c].demand for c in rj)
        if load_i + load_j <= model.capacity:
            # merge: append rj to ri
            routes[i] = ri + rj
            used[j] = False
    # collect merged routes
    solution = [routes[i] for i in range(len(routes)) if used[i]]
    # if more routes than vehicles, merge smallest
    while len(solution) > model.vehicles:
        # merge two smallest by combined cost increase
        best_pair = None
        best_inc = None
        for a in range(len(solution)):
            for b in range(a+1, len(solution)):
                ra, rb = solution[a], solution[b]
                if sum(model.nodes[c].demand for c in ra) + sum(model.nodes[c].demand for c in rb) > model.capacity:
                    continue
                inc = route_cost(ra+rb, dist) - (route_cost(ra, dist) + route_cost(rb, dist))
                if best_pair is None or inc < best_inc:
                    best_pair = (a, b)
                    best_inc = inc
        if best_pair is None:
            break
        a, b = best_pair
        solution[a] = solution[a] + solution[b]
        del solution[b]
    return solution

# ---------------------- Local Search Moves ----------------------

def two_opt_route(route, dist):
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_move = None
        n = len(route)
        for i in range(n-1):
            for j in range(i+2, n):
                # calculate delta of reversing route[i+1:j]
                a, b = route[i], route[i+1]
                c, d = route[j], route[j+1] if j+1<n else None
                # cost before: a->b + c->d (d is depot if None)
                cost_before = dist[a][b] + (dist[c][d] if d is not None else dist[c][0])
                # cost after: a->c_rev + b_rev->d
                cost_after = dist[a][c] + (dist[b][d] if d is not None else dist[b][0])
                delta = cost_after - cost_before
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i+1, j)
        if best_move:
            i1, j1 = best_move
            route[i1:j1+1] = reversed(route[i1:j1+1])
            improved = True
    return route


def local_search(solution, model, dist):
    # Repeatedly apply 2-opt on each route
    improved = True
    while improved:
        improved = False
        # Intra-route 2-opt
        for idx, rt in enumerate(solution):
            old_cost = route_cost(rt, dist)
            new_rt = two_opt_route(rt, dist)
            new_cost = route_cost(new_rt, dist)
            if new_cost < old_cost:
                solution[idx] = new_rt
                improved = True
        # Inter-route relocate
        # Omitted for brevity, can be added later
    return solution

# ---------------------- Perturbation for ILS ----------------------

def perturb(solution, model):
    # Remove one random customer from two random routes and reinsert elsewhere randomly
    sol = [list(r) for r in solution]
    all_routes = list(range(len(sol)))
    for _ in range(2):
        if len(sol) < 2:
            break
        r = random.choice(all_routes)
        if not sol[r]:
            continue
        c = random.choice(sol[r])
        sol[r].remove(c)
        # choose random route to insert
        ri = random.choice(all_routes)
        pos = random.randrange(len(sol[ri])+1)
        sol[ri].insert(pos, c)
    return sol

# ---------------------- Iterated Local Search ----------------------

def iterated_local_search(initial_sol, model, dist, time_limit=240):
    start_time = time.time()
    best = [list(r) for r in initial_sol]
    best_cost = solution_cost(best, dist)
    current = [list(r) for r in initial_sol]
    while time.time() - start_time < time_limit:
        # Perturbation
        pert = perturb(current, model)
        # Local search
        ls = local_search(pert, model, dist)
        ls_cost = solution_cost(ls, dist)
        if ls_cost < best_cost:
            best, best_cost = ls, ls_cost
            current = [list(r) for r in ls]
        else:
            current = [list(r) for r in ls]
    return best

# ---------------------- Output Solution ----------------------

def write_solution(routes, dist, out_file):
    total = solution_cost(routes, dist)
    with open(out_file, 'w') as f:
        f.write(f"TotalCost {total}\n")
        for r in routes:
            line = ' '.join(str(c) for c in r)
            f.write(line + '\n')

# ---------------------- Main ----------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 fcvrp_solver.py <instance_file> <solution_output_file>")
        sys.exit(1)
    inst_file = sys.argv[1]
    out_file = sys.argv[2]
    model = load_model(inst_file)
    dist = compute_distance_matrix(model)
    selected = select_initial_customers(model)
    init_routes = construct_initial_routes(model, dist, selected)
    init_routes = local_search(init_routes, model, dist)
    best = iterated_local_search(init_routes, model, dist)
    write_solution(best, dist, out_file)
    print("Finished. Best cost:", solution_cost(best, dist))

if __name__ == '__main__':
    main()
