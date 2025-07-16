from munkres import Munkres
import json
from create_harg import HARG, Node, Edge

# PERSON 1:
#   UBC: 2
#   LBC: 3
#   GENDER: 1

# PERSON 2:
#   UBC: 5
#   LBC: 3
#   GENDER: 0

# cost for this PERSON object using EPL vertex = 
# RCOST(UBC) because 2 ≠ 5
# + 0 for LBC because 3 == 3
# + RCOST(GENDER) because 1 ≠ 0
############ BELOW FUNCTION does it ####################
def compute_node_cost(node1, node2, rcost_dict):
    """
    Computes replacement cost between two nodes based on properties.
    """
    cost = 0
    for prop in node1.properties:
        val1 = node1.properties.get(prop, None)
        val2 = node2.properties.get(prop, None)
        if val1 != val2:
            cost += rcost_dict.get(prop, 1)  # Default cost if prop not in rcost_dict
    return cost

def build_cost_matrix(graph1_nodes, graph2_nodes, rcost_dict, icost):
    """
    Builds full cost matrix for Munkres alignment.
    Includes insertion/deletion costs.
    """
    n = len(graph1_nodes)
    m = len(graph2_nodes)
    size = max(n, m)
    
    cost_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            if i < n and j < m:
                cost = compute_node_cost(graph1_nodes[i], graph2_nodes[j], rcost_dict)
            else:
                cost = icost
            row.append(cost)
        cost_matrix.append(row)
    return cost_matrix

def calculate_ced_munkres(harg1, harg2, rcost_dict, icost):
    """
    Calculates CED between two HARGs using Munkres assignment.
    """
    # Extract leaf nodes (Entity-Property-Level)
    nodes1 = list(harg1.nodes.values())
    nodes2 = list(harg2.nodes.values())
    
    cost_matrix = build_cost_matrix(nodes1, nodes2, rcost_dict, icost)
    
    m = Munkres()
    indexes = m.compute(cost_matrix)
    
    total_cost = 0
    for row, column in indexes:
        total_cost += cost_matrix[row][column]
    
    return total_cost

if __name__ == "__main__":
    RCOST = {'GENDER': 3, 'UBC': 2, 'LBC': 2}
    ICOST = 2

    with open("dataset/071420250000/hargs/gold/train/2.json", "r") as f:
        graph_data = json.load(f)
    harg1 = HARG.from_dict(graph_data)

    with open("dataset/071420250000/hargs/gold/train/3.json", "r") as f:
        graph_data = json.load(f)
    harg2 = HARG.from_dict(graph_data)

    ced = calculate_ced_munkres(harg1, harg2, RCOST, ICOST)
    print("CED:", ced)


    # basically for MUQNOL, I don't need to use Munkres and HARG to calculate CED,
    # I have    
    #           graph from gold properties
            #   graph from noisy properties                     --  use as training graph   (for FemmIR)
            #   ground_dist_matrix between a pair of graphs     --  use as training label   (for FemmIR)

        # nodes1 = harg1['labels']
        # nodes2 = harg2['labels']