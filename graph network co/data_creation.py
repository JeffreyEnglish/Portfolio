# Solve individual datasets using Gurobi
import gurobipy as gp
import numpy as np
import networkx as nx
import dgl
import tqdm
import torch

def solve_with_Gurobi(data, env):
    """
    Finds a maximum independent set for a given networkx graph using a MIP formulation
    """
    edge_tuples = data.edges

    with gp.Model(env=env) as model:
        x = model.addVars(range(len(data.nodes)), name='x', vtype=gp.GRB.BINARY)
        weights = nx.get_node_attributes(data, 'weight')
        _edge_constraints = model.addConstrs((x[int(a1)] + x[int(a2)] <= 1) for a1,a2 in edge_tuples)

        model.setObjective(gp.quicksum(x[k]*weights[k] for k in range(len(data.nodes))), sense=gp.GRB.MAXIMIZE)
        model.optimize()
        output = {k:int(v.x>1e-3) for k,v in x.items()}
        mip_gap = model.MIPGap

    return output, mip_gap

def solve_with_networkx(graph):
    """
    Finds a maximal indepdent set for a given networkx graph
    """
    maximal_set = nx.maximal_independent_set(graph, seed=1)
    return {k:1 if k in maximal_set else 0 for k in graph.nodes}

def create_er_dataset(N, avg_nodes, edge_probability_mean=0.5, edge_probability_scale=0.2, 
                      seed=1, solver='gurobi', solve_time=20, weighted=True,
                      weight_mean=10, weight_scale=5):
    """
    Generate a dataset of Erdos-Renyi graphs and their solutions.

    Parameters:
    - N (int): Number of graphs to generate.
    - avg_nodes (float): Average number of nodes in each graph (Poisson distributed).
    - edge_probability_mean (float, optional): Mean probability for edge creation in Erdos-Renyi graph generation.
      Defaults to 0.5.
    - edge_probability_scale (float, optional): Scale parameter for the normal distribution of edge probability.
      Defaults to 0.2.
    - seed (int, optional): Seed for random number generation. Defaults to 1.
    - solver (str, optional): Solver to use for solving the graph problems. Supported values are 'gurobi' and 'networkx'.
      Defaults to 'gurobi'.
    - solve_time (int, optional): Time limit in seconds for the solver. Defaults to 20. Only used when solver is 'gurobi'
    - weighted (boolean, optional): Flag if the generated graphs should be weighted (True) or unweighted (False). 
    Defaults to True.
    - weight_mean (float, optional): Mean of the normal distribution of node weights. Defaults to 10.
    - weight_scale (float, optional): Scale parameter for the normal distribution of node weights. Defaults to 5.

    Returns:
    - graphs (dict): Dictionary where keys are integers from 0 to N-1 representing each graph, and values are dictionaries
      containing:
      - 'parameters': Dictionary with parameters used to generate the graph.
      - 'mip_gap' (if solver is 'gurobi'): Mixed Integer Programming gap from Gurobi solver.
      - 'solution': Tensor containing the solution vector for the graph problem.
      - 'dgl_graph': DGL graph object converted from NetworkX graph, with node attributes 'weight' and 'y'.
    """

    np.random.seed(seed)
    graphs = {}
    for p in tqdm.trange(N):
        graphs[p] = {}

        node_count = np.random.poisson(avg_nodes)
        probability = np.random.normal(loc=edge_probability_mean, scale=edge_probability_scale)
        G = nx.erdos_renyi_graph(node_count, probability, seed=p)

        if weighted:
            node_weights = np.random.normal(loc=edge_probability_mean, scale=edge_probability_scale, size=node_count).astype(np.float32)
        else:
            node_weights = np.ones(node_count, dtype=np.float32)
        
        nx.set_node_attributes(G, {k:node_weights[k] for k in G.nodes}, 'weight')
        graphs[p]['parameters'] = {'node_count': node_count,
                                       'probability': probability,
                                       'type': 'erdos-renyi',
                                       'seed': p}

        # Solve
        if solver =='gurobi':
            with gp.Env(params={'OutputFlag': 0, 'TimeLimit': solve_time}) as env:
                output_dict, mip_gap = solve_with_Gurobi(G, env)
                graphs[p]['mip_gap'] = mip_gap
        elif solver=='networkx':
            output_dict = solve_with_networkx(G)
        output_tensor = torch.tensor([output_dict[k] for k in range(node_count)])
        graphs[p]['solution'] = output_tensor
        graphs[p]['mwis'] = (output_tensor.numpy() * node_weights).sum().item()

        # Create DGL version
        nx.set_node_attributes(G, 1.0, 'x')
        nx.set_node_attributes(G, output_dict, 'y')
        dglG = dgl.from_networkx(G, node_attrs=['weight', 'x', 'y'])
        dglG = dgl.add_self_loop(dglG)
        graphs[p]['dgl_graph'] = dglG

    return graphs