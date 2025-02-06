# views.py

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting

import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse
import io
import base64
from scipy.optimize import linprog
import logging
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from scipy.optimize import linprog
import logging
# Configure logging
logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'solver/home.html')

def graphical_solve(request):
    if request.method == 'POST':
        try:
            # Get optimization type
            opt_type = request.POST.get('opt_type', 'maximize')
            c = [float(x) for x in request.POST.get('objective', '').split()]
            if len(c) != 2:
                raise ValueError("Graphical method supports only two variables.")

            constraints = request.POST.getlist('constraints[]')
            A_ub, b_ub = [], []

            for constraint in constraints:
                parts = constraint.strip().split()
                if len(parts) != 3:
                    raise ValueError("Each constraint must have exactly two coefficients and a RHS value.")
                A_ub.append([float(parts[0]), float(parts[1])])
                b_ub.append(float(parts[2]))

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)

            # For maximization, convert to minimization by multiplying c by -1
            if opt_type == 'maximize':
                c_lp = [-ci for ci in c]
            else:
                c_lp = c

            # Set bounds for variables (non-negative variables)
            x_bounds = (0, None)  # (min, max) for all variables

            # Solve LP using linprog
            res = linprog(c=c_lp, A_ub=A_ub, b_ub=b_ub, bounds=[x_bounds, x_bounds], method='highs')

            if res.success:
                x1_opt, x2_opt = res.x
                if opt_type == 'maximize':
                    objective_value = -res.fun  # Multiply by -1 because we minimized
                else:
                    objective_value = res.fun

                optimal_solution = {
                    'x1': round(x1_opt, 4),
                    'x2': round(x2_opt, 4),
                    'objective_value': round(objective_value, 4)
                }
                res_message = None
            else:
                optimal_solution = None
                res_message = res.message

            # Plot constraints and feasible region
            fig, ax = plt.subplots()
            x1_vals = np.linspace(0, max(b_ub) * 1.1, 400)

            for i in range(len(A_ub)):
                a1, a2 = A_ub[i]
                if a2 != 0:
                    x2_vals = (b_ub[i] - a1 * x1_vals) / a2
                    x2_vals = np.maximum(0, x2_vals)  # x2 >= 0
                    ax.plot(x1_vals, x2_vals, label=f'Constraint {i+1}')
                else:
                    x = b_ub[i] / a1
                    ax.axvline(x=x, label=f'Constraint {i+1}')

            # Shade feasible region
            x1_grid = np.linspace(0, max(b_ub)*1.1, 200)
            x2_grid = np.linspace(0, max(b_ub)*1.1, 200)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            Z = np.ones_like(X1, dtype=bool)

            for i in range(len(A_ub)):
                a1, a2 = A_ub[i]
                Z = np.logical_and(Z, (a1 * X1 + a2 * X2 <= b_ub[i]))

            Z = np.logical_and(Z, X1 >= 0)
            Z = np.logical_and(Z, X2 >= 0)

            ax.contourf(X1, X2, Z, levels=[0.5, 1], colors=['#a0ffa0'], alpha=0.3)

            # Plot optimal solution point if available
            if optimal_solution:
                ax.plot(optimal_solution['x1'], optimal_solution['x2'], 'ro', label='Optimal Solution')
                ax.annotate('Optimal Solution', xy=(optimal_solution['x1'], optimal_solution['x2']),
                            xytext=(optimal_solution['x1'] + 0.5, optimal_solution['x2'] + 0.5),
                            arrowprops=dict(facecolor='black', shrink=0.05))

            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title('Graphical Solution')
            ax.legend()
            ax.grid(True)

            # Save plot to in-memory file
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            graph = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            # Render result template with context
            return render(request, 'solver/graphical_result.html', {
                'graph': graph,
                'optimal_solution': optimal_solution,
                'error': res_message
            })

        except Exception as e:
            # Log the exception
            logger.error(f"Exception in graphical_solve view: {e}", exc_info=True)
            # Also print it out to the console (optional)
            print(f"Exception occurred: {e}")
            return render(request, 'solver/graphical_solve.html', {'error': str(e)})

    return render(request, 'solver/graphical_solve.html')

def simplex(c, A, b, maximize=True):
    """
    Implements the Simplex Method for Linear Programming.
    :param c: Coefficients of the objective function.
    :param A: Constraint coefficients matrix.
    :param b: Right-hand side constants of constraints.
    :param maximize: Boolean flag for maximizing (True) or minimizing (False).
    :return: Optimal solution dictionary and optimal value.
    """
    num_constraints, num_variables = A.shape
    slack_vars = np.eye(num_constraints)  

    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))
    obj_row = np.hstack(((-1 if maximize else 1) * c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))

    while True:
        if all(tableau[-1, :-1] >= 0): 
            break

        pivot_col = np.argmin(tableau[-1, :-1])  
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf

        if np.all(ratios == np.inf):
            raise ValueError("The problem is unbounded.")

        pivot_row = np.argmin(ratios)
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        basic_var_index = np.where(tableau[i, :-1] == 1)[0]
        if len(basic_var_index) == 1 and basic_var_index[0] < num_variables:
            solution[basic_var_index[0]] = tableau[i, -1]

    optimal_value = tableau[-1, -1]

    return {f"x{i+1}": solution[i] for i in range(num_variables)}, optimal_value

def simplex_solver(request):
    if request.method == 'POST':
        try:
            maximize = request.POST.get('type') == 'maximize'
            num_variables = int(request.POST.get('variables'))
            c = np.array([float(x) for x in request.POST.get('objective').split()])

            constraints = request.POST.getlist('constraints[]')
            A, b = [], []

            for constraint in constraints:
                parts = constraint.split()
                if len(parts) < num_variables + 1:
                    raise ValueError("Each constraint must have enough coefficients.")
                
                A.append([float(x) for x in parts[:-1]])
                b.append(float(parts[-1]))

            A = np.array(A)
            b = np.array(b)

            solution, optimal_value = simplex(c, A, b, maximize)

            return render(request, 'solver/simplex_result.html', {
                'solution': solution,
                'optimal_value': optimal_value
            })

        except ValueError as e:
            return render(request, 'solver/simplex_result.html', {'error': str(e)})

    return render(request, 'solver/simplex_solve.html')

def transportation_solver(request):
    if request.method == 'POST':
        try:
            # Get number of suppliers and consumers
            num_suppliers = int(request.POST.get('num_suppliers'))
            num_consumers = int(request.POST.get('num_consumers'))

            # Get supply values
            supply = [float(request.POST.get(f'supply_{i}')) for i in range(num_suppliers)]
            # Get demand values
            demand = [float(request.POST.get(f'demand_{j}')) for j in range(num_consumers)]

            # Get cost matrix
            costs = []
            for i in range(num_suppliers):
                row = [float(request.POST.get(f'cost_{i}_{j}')) for j in range(num_consumers)]
                costs.append(row)

            # Convert lists to numpy arrays
            supply = np.array(supply)
            demand = np.array(demand)
            costs = np.array(costs)

            # Check if the problem is balanced
            total_supply = np.sum(supply)
            total_demand = np.sum(demand)
            if total_supply != total_demand:
                raise ValueError("Total supply must equal total demand for a balanced transportation problem.")

            # Formulate the linear programming problem
            # Flatten the cost matrix
            c = costs.flatten()
            # Create the inequality constraints matrix and vector
            A_eq = []
            b_eq = []

            # Supply constraints
            for i in range(num_suppliers):
                constraint = [0] * num_suppliers * num_consumers
                for j in range(num_consumers):
                    constraint[i * num_consumers + j] = 1
                A_eq.append(constraint)
                b_eq.append(supply[i])

            # Demand constraints
            for j in range(num_consumers):
                constraint = [0] * num_suppliers * num_consumers
                for i in range(num_suppliers):
                    constraint[i * num_consumers + j] = 1
                A_eq.append(constraint)
                b_eq.append(demand[j])

            # Bounds for all variables (xij >= 0)
            x_bounds = [(0, None)] * num_suppliers * num_consumers

            # Solve the LP
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

            if res.success:
                # Extract the solution
                flows = res.x.reshape((num_suppliers, num_consumers))
                total_cost = res.fun

                return render(request, 'solver/transportation_result.html', {
                    'flows': flows,
                    'total_cost': total_cost,
                    'num_suppliers': num_suppliers,
                    'num_consumers': num_consumers,
                    'supply': supply,
                    'demand': demand,
                })
            else:
                error_message = res.message
                return render(request, 'solver/transportation_solve.html', {'error': error_message})

        except Exception as e:
            # Log the exception
            logger.error(f"Exception in transportation_solver view: {e}", exc_info=True)
            return render(request, 'solver/transportation_solve.html', {'error': str(e)})

    return render(request, 'solver/transportation_solve.html')