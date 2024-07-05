import time
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import dgl
import tqdm
import numpy as np
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats=1, hidden_size=4, num_classes=2):
        super().__init__()
        self.c1 = GraphConv(in_feats, hidden_size)
        self.c2 = GraphConv(hidden_size, hidden_size)
        self.c3 = GraphConv(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, num_classes)

    def forward(self, g, in_feat):
        # Obtain graph embedding
        x = self.c1(g, in_feat)
        x = nn.ReLU()(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.c2(g, x)
        x = nn.ReLU()(x)
        x = self.c3(g, x)

        # Final classifier
        x = self.linear1(x)
                
        return nn.Softmax(dim=1)(x)

class MISSolution():
    def __init__(self, dgl_graph):
        """
        Initialize the MISSolution object.

        Parameters:
        - dgl_graph (dgl.DGLGraph): Input DGL graph representing the problem instance.
        """
        self.graph = dgl_graph
        self.weights = torch.stack([dgl_graph.ndata['weight'], dgl_graph.ndata['x']], axis=1)
        self.graph.ndata['ts_label'] = torch.zeros_like(self.graph.ndata['weight'])
        self.size = self.graph.num_nodes()
        self.avg_degree = self.graph.num_edges() / self.graph.num_nodes()

    def assign_node(self, node):
        """
        Assign a node in the graph and update its neighbors' labels.

        Parameters:
        - node (int): Index of the node to be assigned.
        """
        for neighbor in self.graph.successors(node):
            self.graph.ndata['ts_label'][neighbor] = -1      
        self.graph.ndata['ts_label'][node] = 1      

    def get_partial_graph(self):
        """
        Get the residual subgraph and unassigned nodes based on current node labels.

        Returns:
        - residual_dgl_graph (dgl.DGLGraph): Residual subgraph with only assigned nodes and their neighbors.
        - unassigned_nodes (list): List of indices of nodes that are not assigned.
        """
        node_labels = self.graph.ndata['ts_label']
        assigned_nodes = torch.argwhere(node_labels).reshape(-1)
        unassigned_nodes = [k for k in range(self.size) if k not in assigned_nodes]
        residual_dgl_graph = copy.deepcopy(self.graph)
        to_remove = residual_dgl_graph.filter_nodes(lambda x: x.data['ts_label'] != 0)
        residual_dgl_graph.remove_nodes(to_remove)
        return residual_dgl_graph, unassigned_nodes

    def search(self, network, device, passes=2):
        """
        Perform a search process to find a solution using a given neural network.

        Parameters:
        - network (torch.nn.Module): Neural network used to predict likelihood of nodes being in the independent set.
        - device (string): Device to be used for inference on the network.
        - passes (int, optional): Number of search passes. Defaults to 2.

        Returns:
        - return_items (list): List of tuples containing (new_solution, is_incomplete), where:
          - new_solution (MISSolution): Updated solution object after each pass.
          - is_incomplete (bool): Flag indicating if the solution is incomplete after the pass.
        """
        return_items = [] # Can replace with fixed size collection
        residual_graph, residual_nodes = self.get_partial_graph()
        new_solution = copy.deepcopy(self)

        for _ in range(passes):
            embeddings = torch.stack([residual_graph.ndata['weight'],residual_graph.ndata['x']], axis=1)
            predictions = -network(residual_graph.to(device), 
                                   embeddings.to(device)
                                   )[:,1].cpu().detach().numpy()
            
            # Add some noise to avoid always taking the same path through the tree
            predictions = predictions * np.random.normal(loc=1.0, scale=0.15, size=len(predictions))
            node_order = np.argsort(predictions)

            for idx in node_order:
                node = residual_nodes[idx]
                if new_solution.graph.ndata['ts_label'][node] != 0:
                    break
                new_solution.assign_node(node)

            labels = new_solution.graph.ndata['ts_label']
            is_incomplete = np.sum([x != 0 for x in labels]) < new_solution.size
            return_items.append((new_solution, is_incomplete))

        return return_items
    
    def check_solution_feasibility(self):
        """
        Check the feasibility of the current solution.

        Returns:
        - feasible (bool): True if the solution is feasible (no edges between nodes with label 1), False otherwise.
        """
        graph = copy.deepcopy(self.graph)
        to_remove = graph.filter_nodes(lambda x: x.data['ts_label'] != 1)
        graph.remove_nodes(to_remove)
        graph = dgl.remove_self_loop(graph)
        feasible = graph.num_edges() == 0
        return feasible

    def partial_score(self):
        """
        Calculate the partial score of the current solution. Defined as the number of nodes in the set divided by the total number of assigned nodes

        Returns:
        - score (float): Partial score based on the number of positive labels divided by the number of assigned nodes plus one.
        """
        mis = np.sum([x>0 for x in self.graph.ndata['ts_label']])
        assigned = len(torch.argwhere(self.graph.ndata['ts_label']).reshape(-1))
        return mis/(assigned+1)

def mis_search(graph, network, time_limit, passes=2, verbose=False, p_best=False):
    """
    Perform a search for a Maximum Independent Set (MIS) solution with diversity using a given neural network.

    Parameters:
    - graph (dgl.DGLGraph): Input graph for which the MIS solution is sought.
    - network (torch.nn.Module): Neural network used to predict node importance.
    - time_limit (float): Maximum time limit (in seconds) for the search process.
    - passes (int, optional): Number of search passes per iteration. Defaults to 2.
    - verbose (bool, optional): If True, print progress information during the search. Defaults to False.
    - p_best (int or False, optional): If provided, select the best p_best solutions to expand in each iteration.
      Defaults to False (select a solution randomly).

    Returns:
    - max_mis (int): Size of the maximum independent set found.
    - best_solution (MISSolution or None): Best solution found as an instance of MISSolution class.
    - progress (dict): Dictionary tracking the progress of the search with keys as iteration numbers and values as dictionaries
      containing:
      - 'mis' (int): Size of the maximum independent set at that iteration.
      - 'queue' (int): Number of solutions in the queue waiting to be expanded.
      - 'time' (float): Time elapsed since the start of the search at that iteration.
    """

    # Initialize values
    progress = {}
    max_mwis = -np.inf
    best_solution = None
    t0 = time.time()
    queue = [MISSolution(graph)]
    score_list = [queue[0].partial_score()]
    loops = 0

    # Move everything to the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = network.to(device)
    graph = graph.to(device)

    while (time.time()-t0 < time_limit) and len(queue)>0:
        # Choose a solution from the queue to expand
        if p_best:
            # Choose the index of one of the p best solutions based on score
            pb = min(len(queue), p_best)
            mis_index = random.randrange(pb)
            index = np.argsort(-np.array(score_list))[mis_index] 
        else:
            # Choose a random solution index
            index = random.randrange(len(queue)) 

        root = queue.pop(index) # Remove the chosen solution from the queue
        _ = score_list.pop(index) # Remove the corresponding score from score_list

        # Expand the chosen solution and get new solutions and their completeness status
        new_items = root.search(network, device, passes=passes)
        for item, incomplete in new_items:
            # Calculate MIS size of the new solution
            mwis = np.sum([item.weights[i] for i,x in enumerate(item.graph.ndata['ts_label']) if x==1])

            # If the solution is incomplete, add it back to the queue with its score
            if incomplete:
                queue.append(item)
                score_list.append(mwis)

            # Update the best solution found if the current solution has a larger MIS
            if mwis > max_mwis:
                max_mwis = mwis
                if verbose:
                    print(time.time()-t0, max_mwis)
                best_solution = item

        # Record progress information for the current iteration
        progress[loops] = {'mwis':max_mwis, 'queue': len(queue), 'time':time.time()-t0}
        loops += 1
        if verbose:
            print(progress[loops-1])
    
    return max_mwis, best_solution, progress

def train_gcn(net, train_gdl, test_gdl, optimizer, epochs):
    """
    Train a Graph Convolutional Network (GCN) on the given datasets. The GCN is used to predict if nodes are in the maximum independent set or not

    Parameters:
    - net (torch.nn.Module): The GCN model to be trained.
    - train_gdl (dgl.dataloading.GraphDataLoader): DataLoader for the training dataset.
    - test_gdl (dgl.dataloading.GraphDataLoader): DataLoader for the test dataset.
    - optimizer (torch.optim.Optimizer): Optimizer to use during training (e.g., Adam, SGD).
    - epochs (int): Number of training epochs.

    Returns:
    - train_loss (np.ndarray): Array of training losses for each epoch.
    - test_loss (np.ndarray): Array of test losses for each epoch.
    - test_acc (np.ndarray): Array of test accuracies (percentage) - how many nodes are correctly assigned - for each epoch.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    loop = tqdm.trange(epochs)

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    test_acc = np.zeros(epochs)

    loop = tqdm.trange(epochs)
    for epoch in loop:
        net.train()
        for G in train_gdl:
            optimizer.zero_grad()
            graph = G.to(device).to(device)
            embedding = torch.stack([graph.ndata['weight'],graph.ndata['x']], axis=1).to(device)
            y = graph.ndata['y']

            out = net(graph, embedding)

            # Weight positive and negative classes equally
            # Helps avoid local minimum at all False
            positive_density = (y.sum()/len(y)).item()
            class_weights = torch.tensor([1.0, 1.0/positive_density]).to(device)
            loss = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)(out, y)
            train_loss[epoch] += loss.sum().to('cpu').item()
            loss.backward()
            optimizer.step()

        train_loss[epoch] /= sum([x.num_nodes() for x in train_gdl.dataset])
        correct, trials = 0, 0
        net.eval()
        for G in test_gdl:
            graph = G.to(device)
            embedding = torch.stack([graph.ndata['weight'],graph.ndata['x']], axis=1).to(device)
            y = graph.ndata['y']

            out = net(graph, embedding)
            loss = nn.CrossEntropyLoss(reduction='sum')(out, y)
            test_loss[epoch] += loss.sum()
            pred = torch.argmax(out, dim=1)
            correct += int((pred == y).sum())
            trials += len(pred)

        test_loss[epoch] /= sum([x.num_nodes() for x in test_gdl.dataset])
        test_acc[epoch] = correct / trials
        
        loop.set_description(f"Epoch {epoch} - Train Loss: {train_loss[epoch]:.3f} - Test Loss {test_loss[epoch]:.3f}")
