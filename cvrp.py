import argparse
import os
import random
from Parser import load_model
from SolutionValidator import validate_solution

def generate_random_solution(model, output_file):
    """
    Generate a random solution for the CVRP problem.
    Each family will be visited the required number of times.
    
    Args:
        model: The CVRP model
        output_file: The file to write the solution to
    
    Returns:
        routes: List of routes, where each route is a list of node IDs
    """
    # Collect nodes to visit based on required visits
    nodes_to_visit = []
    for family in model.families:
        # Get random members from the family for each required visit
        for _ in range(family.required_visits):
            if family.nodes:
                random_node = random.choice(family.nodes)
                nodes_to_visit.append(random_node.id)

    # Shuffle the nodes to make it random
    random.shuffle(nodes_to_visit)
    
    # Create routes
    routes = []
    for _ in range(model.vehicles):
        if not nodes_to_visit:
            break
            
        route = [0]  # Start at depot
        load = 0
        
        # Keep adding nodes until capacity is reached
        while nodes_to_visit and load < model.capacity:
            node_id = nodes_to_visit[0]
            node = model.nodes[node_id]
            
            if load + node.demand <= model.capacity:
                route.append(node_id)
                load += node.demand
                nodes_to_visit.pop(0)
            else:
                break
                
        route.append(0)  # Return to depot
        routes.append(route)
    
    # If there are remaining nodes, add them to the last route
    # (This is just to ensure all required visits are included in the solution)
    if nodes_to_visit:
        last_route = routes[-1]
        for node_id in nodes_to_visit:
            last_route.insert(-1, node_id)
    
    # Write routes to output file
    with open(output_file, 'w') as f:
        for route in routes:
            f.write(' '.join(map(str, route)) + '\n')
    
    return routes

def solve_cvrp(instance_file, output_file):
    """
    Solve the CVRP problem and write the solution to a file
    
    Args:
        instance_file: The CVRP instance file
        output_file: The file to write the solution to
    
    Returns:
        The output file path
    """
    # Load the model
    model = load_model(instance_file)
    
    # Generate a random solution
    routes = generate_random_solution(model, output_file)
    
    # Validate the solution
    valid, report = validate_solution(model, routes)
    
    if valid:
        print(f"Generated a valid solution with total cost: {report['total_cost']}")
    else:
        print("Generated solution is invalid. Errors:")
        for error in report['errors']:
            print(f"- {error}")
    
    print(f"Solution written to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='CVRP Solver')
    parser.add_argument('instance_file', help='Path to the CVRP instance file')
    parser.add_argument('--output', '-o', default='solution.txt', help='Path to write the solution file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.instance_file):
        print(f"Error: Instance file '{args.instance_file}' does not exist.")
        return
    
    solve_cvrp(args.instance_file, args.output)

if __name__ == "__main__":
    main()
