import argparse
import os
import random
from collections import deque
from Parser import load_model
from SolutionValidator import validate_solution
import math

def route_cost(route, cost_matrix):
    return sum(cost_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

def generate_cw_solution(model, seed=None):
    """
    Solve the CVRP using Clarke–Wright savings heuristic.
    """
    if seed is not None:
        random.seed(seed)
    visits = []
    for fam in model.families:
        if fam.required_visits > len(fam.nodes):
            raise ValueError
        visits.extend(random.sample([n.id for n in fam.nodes], fam.required_visits))
    Q, C = model.capacity, model.cost_matrix
    routes = {i: [0,i,0] for i in visits}
    loads = {i: model.nodes[i].demand for i in visits}
    savings = sorted(
        [(C[0][i]+C[0][j]-C[i][j],i,j) for i in visits for j in visits if i<j],
        key=lambda x: x[0], reverse=True
    )
    for s,i,j in savings:
        ri = next((k for k,v in routes.items() if v[1]==i or v[-2]==i), None)
        rj = next((k for k,v in routes.items() if v[1]==j or v[-2]==j), None)
        if ri is None or rj is None or ri==rj: continue
        if loads[ri]+loads[rj] > Q: continue
        ri_r, rj_r = routes[ri], routes[rj]
        if ri_r[-2]==i and rj_r[1]==j:
            new = ri_r[:-1]+rj_r[1:]
        elif rj_r[-2]==j and ri_r[1]==i:
            new = rj_r[:-1]+ri_r[1:]
        else:
            continue
        routes[ri], loads[ri] = new, loads[ri]+loads[rj]
        del routes[rj], loads[rj]
        if len(routes)<=model.vehicles: break
    if len(routes)>model.vehicles:
        flat=[n for rt in routes.values() for n in rt if n]
        new_r=[[] for _ in range(model.vehicles)]; new_l=[0]*model.vehicles
        for n in flat:
            d=model.nodes[n].demand
            for v in range(model.vehicles):
                if new_l[v]+d<=Q:
                    new_r[v].append(n); new_l[v]+=d; break
        return [[0]+r+[0] for r in new_r if r]
    return list(routes.values())

def two_opt(route, cost_matrix):
    best, best_cost = route, route_cost(route, cost_matrix)
    for i in range(1,len(route)-2):
        for j in range(i+1,len(route)-1):
            if j-i==1: continue
            cand = route[:i]+route[i:j+1][::-1]+route[j+1:]
            c = route_cost(cand, cost_matrix)
            if c<best_cost: best, best_cost = cand, c
    return best

def relocate(routes, model):
    """
    Relocate a customer between routes for cost improvement.
    """
    best_gain = 0
    best_move = None
    C = model.cost_matrix
    # compute loads
    loads = [sum(model.nodes[n].demand for n in r if n!=0) for r in routes]
    for r1 in range(len(routes)):
        for idx in range(1, len(routes[r1]) - 1):
            cust = routes[r1][idx]
            demand = model.nodes[cust].demand
            for r2 in range(len(routes)):
                if r2 == r1: continue
                if loads[r2] + demand > model.capacity: continue
                for pos in range(1, len(routes[r2])):
                    # cost before
                    before = (C[routes[r1][idx-1]][cust] + C[cust][routes[r1][idx+1]] +
                              C[routes[r2][pos-1]][routes[r2][pos]])
                    after = (C[routes[r1][idx-1]][routes[r1][idx+1]] +
                             C[routes[r2][pos-1]][cust] + C[cust][routes[r2][pos]])
                    gain = before - after
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (r1, idx, r2, pos, gain)
    if best_move:
        r1, idx, r2, pos, _ = best_move
        cust = routes[r1].pop(idx)
        routes[r2].insert(pos, cust)
        return True
    return False

def local_search(solution,model):
    """
    Apply 2-opt and relocate until no improvement.
    """
    sol = [r[:] for r in solution]
    improved = True
    while improved:
        improved = False
        # 2-opt per route
        for i in range(len(sol)):
            new_rt = two_opt(sol[i], model.cost_matrix)
            if route_cost(new_rt, model.cost_matrix) < route_cost(sol[i], model.cost_matrix):
                sol[i] = new_rt
                improved = True
        # relocate between routes
        if relocate(sol, model):
            improved = True
    return sol

def perturb(solution, model):
    """
    Shake: remove a random segment from one route and insert into another,
    ensuring capacity feasibility by retrying if invalid.
    """
    max_attempts = 10
    for _ in range(max_attempts):
        sol = [r[:] for r in solution]
        if len(sol) < 2: return sol
        r1, r2 = random.sample(range(len(sol)), 2)
        if len(sol[r1]) <= 4: continue
        i = random.randint(1, len(sol[r1]) - 3)
        j = random.randint(i + 1, len(sol[r1]) - 2)
        segment = sol[r1][i:j]
        del sol[r1][i:j]
        positions = list(range(1, len(sol[r2])))
        random.shuffle(positions)
        for pos in positions:
            cand = [r[:] for r in sol]
            for idx,n in enumerate(segment): cand[r2].insert(pos+idx, n)
            def load(r): return sum(model.nodes[n].demand for n in r if n)
            if load(cand[r1])<=model.capacity and load(cand[r2])<=model.capacity:
                return cand
    return solution

def iterated_local_search(model,output_file,seed=None,iter_max=50,time_limit=150):
    import time
    if seed is not None: random.seed(seed)

    # Initial solution via CW + local search
    best = generate_cw_solution(model,seed)
    best = local_search(best,model)
    best_cost = sum(route_cost(r,model.cost_matrix) for r in best)
    curr = [r[:] for r in best]
    start = time.time()
    # Iterated Local Search with time limit
    while time.time() - start < time_limit:
        # perturb + local search
        cand = perturb(curr, model)
        cand = local_search(cand,model)
        c_cost = sum(route_cost(r,model.cost_matrix) for r in cand)
        # accept
        curr = [r[:] for r in cand]
        if c_cost < best_cost:
            best, best_cost = [r[:] for r in cand], c_cost
    # write best
    flat_list = [item for sublist in best for item in sublist]

    with open(output_file, 'w') as f:
        line = []
        zero_count = 0

        for num in flat_list:
            line.append(str(num))
            if num == 0:
                zero_count += 1
                if zero_count == 2:
                    f.write(' '.join(line) + '\n')
                    line = []
                    zero_count = 0

        # Write any remaining items that don't end in two 0s
        if line:
            f.write(' '.join(line) + '\n')
    return best

def solve_cvrp(instance_file,output_file='solution.txt',seed=None):
    model = load_model(instance_file)
    total = sum(f.demand*f.required_visits for f in model.families)
    if total > model.vehicles*model.capacity: raise ValueError
    routes = iterated_local_search(model,output_file,seed)
    valid, report = validate_solution(model, routes)
    if not valid: raise RuntimeError('Invalid:'+','.join(report['errors']))
    print(f"Valid solution with total cost: {report['total_cost']}")
    return output_file

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('instance_file')
    p.add_argument('-o','--output',default='solution.txt')
    p.add_argument('-s','--seed',type=int)
    a=p.parse_args()
    if not os.path.exists(a.instance_file): print('Instance not found'); exit(1)
    solve_cvrp(a.instance_file,a.output,a.seed)
