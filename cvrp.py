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
        reverse=True
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
            for v in range(model.vehicles):
                d=model.nodes[n].demand
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

def perturb(solution, model):
    """
    Shake: remove a random segment from one route and insert into another,
    ensuring capacity feasibility by retrying if invalid.
    """
    max_attempts = 10
    for _ in range(max_attempts):
        sol = [r[:] for r in solution]
        # choose two distinct routes
        if len(sol) < 2:
            return sol
        r1, r2 = random.sample(range(len(sol)), 2)
        if len(sol[r1]) <= 4:
            continue
        # select random segment in r1
        i = random.randint(1, len(sol[r1]) - 3)
        j = random.randint(i + 1, len(sol[r1]) - 2)
        segment = sol[r1][i:j]
        # remove segment
        del sol[r1][i:j]
        # try insertion positions in r2
        insertion_positions = list(range(1, len(sol[r2])))
        random.shuffle(insertion_positions)
        for pos in insertion_positions:
            candidate = [r[:] for r in sol]
            for idx, n in enumerate(segment):
                candidate[r2].insert(pos + idx, n)
            # check capacity for both routes
            def load(route):
                return sum(model.nodes[n].demand for n in route if n != 0)
            if load(candidate[r1]) <= model.capacity and load(candidate[r2]) <= model.capacity:
                return candidate
        # if no valid insertion, retry
    # fallback: return original
    return solution

def local_search(solution,model):
    """
    Apply 2-opt on all routes until no improve.
    """
    improved=True; sol=[r[:] for r in solution]
    while improved:
        improved=False
        for idx,rt in enumerate(sol):
            new=two_opt(rt,model.cost_matrix)
            if route_cost(new,model.cost_matrix)<route_cost(rt,model.cost_matrix):
                sol[idx]=new; improved=True
    return sol

def iterated_local_search(model,output_file,seed=None,iter_max=50):
    if seed is not None: random.seed(seed)
    best=generate_cw_solution(model,seed)
    best=local_search(best,model)
    best_cost=sum(route_cost(r,model.cost_matrix) for r in best)
    curr, curr_cost = best, best_cost
    for _ in range(iter_max):
        cand = perturb(curr, model)
        cand=local_search(cand,model)
        c_cost=sum(route_cost(r,model.cost_matrix) for r in cand)
        if c_cost<best_cost:
            best, best_cost = cand, c_cost
            curr, curr_cost = cand, c_cost
        else:
            curr, curr_cost = cand, c_cost
    # write
    with open(output_file,'w') as f:
        for r in best: f.write(' '.join(map(str,r))+'\n')
    return best

def solve_cvrp(instance_file,output_file,seed=None):
    model=load_model(instance_file)
    total=sum(f.demand*f.required_visits for f in model.families)
    if total>model.vehicles*model.capacity: raise ValueError
    routes=iterated_local_search(model,output_file,seed)
    valid,report=validate_solution(model,routes)
    if not valid: raise RuntimeError('Invalid:'+','.join(report['errors']))
    print(f"Valid solution with total cost: {report['total_cost']}")
    return output_file

if __name__=='__main__':
    p=argparse.ArgumentParser();
    p.add_argument('instance_file');p.add_argument('-o','--output',default='solution.txt');p.add_argument('-s','--seed',type=int)
    a=p.parse_args();
    if not os.path.exists(a.instance_file): print('Instance not found');exit(1)
    solve_cvrp(a.instance_file,a.output,a.seed)
