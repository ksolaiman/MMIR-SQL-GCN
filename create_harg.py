#!/usr/bin/env python3
import os
import json
import pickle
import argparse
import uuid
import utils
from psycopg2.extensions import AsIs

CONFIG_PATH = "config.json"
UNK_TOKEN = "<UNK>"

# === Classes ===

class Node:
    def __init__(self, node_type, label, properties=None):
        self.id = str(uuid.uuid4())
        self.node_type = node_type
        self.label = label
        self.properties = properties or {}

class Edge:
    def __init__(self, source, target, relation):
        self.source = source
        self.target = target
        self.relation = relation

class HARG:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node):
        self.nodes[node.id] = node
        return node.id

    def add_edge(self, src_id, tgt_id, relation):
        self.edges.append(Edge(src_id, tgt_id, relation))

    def to_dict(self):
        return {
            "nodes": [
                {
                    "id": n.id,
                    "node_type": n.node_type,
                    "label": n.label,
                    "properties": n.properties
                } for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation
                } for e in self.edges
            ]
        }
    
    @staticmethod
    def from_dict(data):
        harg = HARG()
        for node_data in data["nodes"]:
            node = Node(node_data["node_type"], node_data["label"], node_data.get("properties", {}))
            node.id = node_data["id"]
            harg.nodes[node.id] = node
        for edge_data in data["edges"]:
            harg.edges.append(Edge(edge_data["source"], edge_data["target"], edge_data["relation"]))
        return harg



# === HARG Building ===

def build_harg_from_properties(sample_properties):
    harg = HARG()
    root = Node(node_type="ROOT", label="ROOT")
    root_id = harg.add_node(root)

    # Metadata
    metadata = sample_properties.get("metadata", {})
    for k, v in metadata.items():
        meta_node = Node("METADATA", k, {"value": v})
        meta_id = harg.add_node(meta_node)
        harg.add_edge(root_id, meta_id, "hasMetadata")

    # Objects
    for obj in sample_properties.get("objects", []):
        obj_node = Node(obj["type"], obj["type"])
        obj_id = harg.add_node(obj_node)
        harg.add_edge(root_id, obj_id, "hasObject")

        for prop_key, prop_value in obj["properties"].items():
            if isinstance(prop_value, list) and prop_key == "CLOTHES":
                for clothes_item in prop_value:
                    clothes_node = Node(
                        "CLOTHES",
                        clothes_item.get("TYPE", "UNKNOWN"),
                        {"COLOR": clothes_item.get("COLOR", "UNKNOWN")}
                    )
                    clothes_id = harg.add_node(clothes_node)
                    harg.add_edge(obj_id, clothes_id, "wears")

            else:
                prop_node = Node(
                    "PROPERTY",
                    prop_key,
                    {"value": prop_value}
                )
                prop_id = harg.add_node(prop_node)
                harg.add_edge(obj_id, prop_id, prop_key)
    return harg

# === Encoding ===

def encode_node_label_auto(node, label_vocab):
    parts = [node.node_type, node.label] if node.label != node.node_type else [node.node_type]
    for k, v in sorted(node.properties.items()):
        parts.append(f"{k}:{v}")
    label_str = "|".join(parts)

    if label_str not in label_vocab:
        label_vocab[label_str] = len(label_vocab)
    return label_vocab[label_str]

def encode_node_label_fixed(node, label_vocab):
    parts = [node.node_type, node.label] if node.label != node.node_type else [node.node_type]
    for k, v in sorted(node.properties.items()):
        parts.append(f"{k}:{v}")
    label_str = "|".join(parts)
    return label_vocab.get(label_str, label_vocab[UNK_TOKEN])

def get_edge_type_id(relation, edge_type_vocab):
    if relation not in edge_type_vocab:
        edge_type_vocab[relation] = len(edge_type_vocab)
    return edge_type_vocab[relation]

def harg_to_simgnn_graph(harg, label_vocab, edge_type_vocab, phase):
    node_list = list(harg.nodes.values())
    index_map = {node.id: idx for idx, node in enumerate(node_list)}

    if phase == "train":
        labels = [encode_node_label_auto(node, label_vocab) for node in node_list]
    else:
        labels = [encode_node_label_fixed(node, label_vocab) for node in node_list]

    edges = []
    edge_types = []
    for edge in harg.edges:
        src_idx = index_map[edge.source]
        tgt_idx = index_map[edge.target]
        edges.append([src_idx, tgt_idx])
        edges.append([tgt_idx, src_idx])

        # added to save edge labels
        edge_type_id = get_edge_type_id(edge.relation, edge_type_vocab)
        edge_types.append(edge_type_id)
        edge_types.append(edge_type_id)

    return {
        "graph": edges,
        "labels": labels,
        "edge_types": edge_types
    }

def save_simgnn_graph_json(graph_dict, output_path, noisy=None):
    with open(output_path, "w") as f:
        json.dump(graph_dict, f, indent=2)

# === Main Phases ===

def process_phase_train(config):
    print("[INFO] Running PHASE: TRAIN")
    dataset_dir = config["dataset_dir"]
    graph_parent_dir = os.path.join(dataset_dir, config.get("simgnn_graph_dir"),
                             "noisy" if config.get('noisy') else "gold")
    graph_dir = os.path.join(graph_parent_dir, "train")
    os.makedirs(graph_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "trainpool.pkl"), "rb") as f:
        train_ids = pickle.load(f)

    label_vocab = {UNK_TOKEN: 0}  # Start with UNK = 0
    edge_type_vocab = {}

    conn = utils.connect_to_database(config)

    for idx in train_ids:
        properties = get_properties_from_MUQNOL(conn, idx, config.get('noisy'))
        if properties is None:
            print(f"[WARN] Skipping ID {idx}")
            continue
        harg = build_harg_from_properties(properties)
        graph_json = harg_to_simgnn_graph(harg, label_vocab, edge_type_vocab, phase="train")
        if config.get('save_simgnn_graphs'):
            save_simgnn_graph_json(graph_json, os.path.join(graph_dir, f"{idx}.json"))
        if config.get('save_hargs'):
            save_simgnn_graph_json(harg.to_dict(), os.path.join(dataset_dir, config.get("harg_dir"),
                             "noisy" if config.get('noisy') else "gold", "train", f"{idx}.json"))

    # Save vocab
    if config.get('save_simgnn_graphs'):
        with open(os.path.join(graph_parent_dir, "label_vocab.json"), "w") as f:
            json.dump(label_vocab, f, indent=2)
        with open(os.path.join(graph_parent_dir, "edge_type_vocab.json"), "w") as f:
            json.dump(edge_type_vocab, f, indent=2)
        print(f"[INFO] Saved label_vocab.json with {len(label_vocab)} entries.")

def process_phase_test(config):
    print("[INFO] Running PHASE: TEST")
    dataset_dir = config["dataset_dir"]
    graph_parent_dir = os.path.join(dataset_dir, config.get("simgnn_graph_dir"),
                             "noisy" if config.get('noisy') else "gold")
    graph_dir = os.path.join(graph_parent_dir, "test")
    os.makedirs(graph_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "testset.pkl"), "rb") as f:
        test_ids = pickle.load(f)

    with open(os.path.join(graph_parent_dir, "label_vocab.json")) as f:
        label_vocab = json.load(f)

    with open(os.path.join(graph_parent_dir,  "edge_type_vocab.json")) as f:
        edge_type_vocab = json.load(f)

    conn = utils.connect_to_database(config)
    for idx in test_ids:
        properties = get_properties_from_MUQNOL(conn, idx, config.get('noisy'))
        if properties is None:
            print(f"[WARN] Skipping ID {idx}")
            continue
        harg = build_harg_from_properties(properties)
        graph_json = harg_to_simgnn_graph(harg, label_vocab, edge_type_vocab, phase="test")
        if config.get('save_simgnn_graphs'):
            save_simgnn_graph_json(graph_json, os.path.join(graph_dir, f"{idx}.json"))
        if config.get('save_hargs'):
            save_simgnn_graph_json(harg.to_dict(), os.path.join(dataset_dir, config.get("harg_dir"),
                             "noisy" if config.get('noisy') else "gold", "train", f"{idx}.json"))
                             
    print("[INFO] Finished TEST graph generation.")


# === MUQNOL specific functions ====

def get_properties_from_MUQNOL(conn, itemID, noisy=True):
    """
    Fetches properties from either mmir_predicted (noisy) or mmir_ground (gold) table
    based on the 'noisy' boolean.
    """
    table_name = "mmir_predicted" if noisy else "mmir_ground" 

    dbcur = conn.cursor()
    sql = dbcur.mogrify("""
        SELECT mgid, ubc, lbc, gender
        FROM %s
        WHERE mgid = %s
    """, (AsIs(table_name), AsIs(itemID)))
    dbcur.execute(sql)
    rows = dbcur.fetchall()
    # return rows[0]

    if rows is None:
        raise ValueError(f"No record found for mgid={itemID}")

    mgid, ubc, lbc, gender = rows[0]        

    # Convert to structured dict
    properties = {
        # "metadata": {
        #     "mgid": mgid
        # },                        # for every item it is different aka adds 1 to CED, no need to add in CED; 
        "objects": [
            {
                "type": "PERSON",
                "properties": {
                    "UBC": ubc,
                    "LBC": lbc,
                    "GENDER": gender
                }
            }
        ]
    }

    return properties



# === Entry point ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "test"], required=True,
                        help="Phase to run: train or test")
    args = parser.parse_args()

    config = utils.load_config()

    if args.phase == "train":
        process_phase_train(config)
    else:
        process_phase_test(config)

if __name__ == "__main__":
    main()
