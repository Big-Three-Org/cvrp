# Authors

- Giannis Tampakis t8220146
- Thodoris Zarkalis t8220043
- Glytsos Dionysios t8220029

# Explanation of our logic

This script is designed to solve the Capacitated Vehicle Routing Problem (CVRP) using a combination of heuristic and local search algorithms. Below is a detailed explanation of how the script works and the rationale behind each algorithm used.

## Overview

The script reads a CVRP instance from a file, generates an initial solution using the Clarke-Wright savings heuristic, and then refines this solution using local search techniques. The final solution is validated and written to an output file.

## Key Components

1. **Clarke-Wright Savings Heuristic**
   - **Function**: `generate_cw_solution`
   - **Purpose**: This heuristic is used to generate an initial feasible solution for the CVRP. It starts by creating a separate route for each customer and then iteratively merges routes based on the savings in cost.
   - **Rationale**: The Clarke-Wright heuristic is a well-known method for quickly generating a good initial solution for vehicle routing problems. It is efficient and provides a solid starting point for further optimization.

2. **2-opt Algorithm**
   - **Function**: `two_opt`
   - **Purpose**: This algorithm is used to improve individual routes by reversing segments of the route to reduce the total travel cost.
   - **Rationale**: The 2-opt algorithm is a simple yet effective local search technique for optimizing routes. It helps in reducing the total distance by eliminating crossovers in the route.

3. **Relocate Operator**
   - **Function**: `relocate`
   - **Purpose**: This operator moves a customer from one route to another to improve the overall cost.
   - **Rationale**: Relocating customers between routes can lead to significant cost savings by balancing the load and reducing travel distances.

4. **Local Search**
   - **Function**: `local_search`
   - **Purpose**: This function applies the 2-opt and relocate operators iteratively until no further improvements can be made.
   - **Rationale**: Local search is a powerful technique for refining solutions. By iteratively applying optimization operators, the solution quality is improved.

5. **Perturbation**
   - **Function**: `perturb`
   - **Purpose**: This function introduces diversity by randomly altering the solution, which helps in escaping local optima.
   - **Rationale**: Perturbation is used to avoid getting stuck in local optima by exploring new areas of the solution space.

6. **Iterated Local Search**
   - **Function**: `iterated_local_search`
   - **Purpose**: This function combines perturbation and local search to iteratively improve the solution.
   - **Rationale**: Iterated local search is a metaheuristic that enhances the exploration of the solution space by alternating between perturbation and local search.

7. **Solution Validation**
   - **Function**: `validate_solution`
   - **Purpose**: This function checks the feasibility of the solution by ensuring all constraints are satisfied.
   - **Rationale**: Validation is crucial to ensure that the solution is not only cost-effective but also feasible according to the problem constraints.

## Experiments and Results

We conducted multiple experiments with different seeds and time limits to evaluate the performance of the algorithm. The best result within the given time limit of 150 seconds was achieved with seed 15, yielding a score of 593. Additionally, we extended the time limit to 30 minutes using the same seed and managed to achieve an exceptional score of 553.

## Conclusion

The combination of these algorithms allows the script to efficiently solve the CVRP by generating a good initial solution and then refining it through local search techniques. The use of perturbation and iterated local search helps in exploring the solution space more effectively, leading to high-quality solutions. 