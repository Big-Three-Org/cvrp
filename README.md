# Capacitated Vehicle Routing Problem (CVRP) Solver

This project implements a solution for the Capacitated Vehicle Routing Problem (CVRP) using a combination of heuristic and local search algorithms. The implementation is designed to efficiently solve complex routing problems while respecting vehicle capacity constraints and customer visit requirements.

## Authors

- Giannis Tampakis (t8220146)
- Thodoris Zarkalis (t8220043)
- Glytsos Dionysios (t8220029)

## Project Overview

The solver uses a sophisticated approach combining multiple algorithms to find optimal or near-optimal solutions for CVRP instances. The implementation includes:

1. **Clarke-Wright Savings Heuristic** for initial solution generation
2. **2-opt Algorithm** for route optimization
3. **Relocate Operator** for inter-route improvements
4. **Local Search** techniques
5. **Perturbation** mechanisms
6. **Iterated Local Search** for global optimization

## Features

- Efficient solution generation using the Clarke-Wright savings heuristic
- Advanced local search techniques for solution refinement
- Solution validation to ensure feasibility
- Support for family-based customer visits
- Vehicle capacity constraints handling
- Comprehensive solution reporting

## Project Structure

- `Main.py` - Main entry point and program execution
- `Parser.py` - Handles input file parsing and model loading
- `SolutionValidator.py` - Validates solution feasibility
- `fcvrp_runner.py` - Core CVRP solving implementation
- `explanation.md` - Detailed explanation of the algorithms and approach

## Usage

1. Ensure you have Python installed on your system
2. Place your CVRP instance file in the project directory
3. Run the program:

```bash
python Main.py
```

By default, the program will process the file `fcvrp_P-n101-k4_10_3_3.txt`. To use a different instance file, modify the file name in `Main.py`.

## Performance

The solver has been tested with various instances and has shown promising results:
- Best result within 150 seconds: Score of 593
- Extended time limit (30 minutes): Score of 553

## Solution Output

The program generates a solution file (`solution.txt`) containing the optimized routes. The solution includes:
- Route assignments for each vehicle
- Total cost of the solution
- Load distribution across vehicles
- Family visit statistics

## Validation

The solution validator ensures that all constraints are satisfied:
- Vehicle capacity limits
- Required customer visits
- Route feasibility
- Family visit requirements

## License

This project is part of an academic assignment and is not licensed for commercial use. 