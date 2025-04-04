from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from finetune.model import GNN_graphpred
from rdkit.Chem import AllChem
import torch
from finetune.loader import mol_to_graph_data_obj_simple
import os
from typing import Optional, List
import torch.nn as nn
import numpy as np
from tqdm import tqdm

MODEL_PATHS = {
    "MGSSL": os.path.join(os.path.dirname(__file__), "motif_based_pretrain/saved_model/pretrained.pth")}

class State:
    model: Optional[GNN_graphpred] = None

    model_name: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[int] = None

    initialized: bool = False

def setup(model_name: str, device: str, batch_size: int) -> None:
    num_layer = 5
    emb_dim = 300
    dropout_ratio = 0.5
    graph_pooling = "mean"
    JK = "last"
    gnn_type = "gin"

    model = GNN_graphpred(num_layer, emb_dim, 1, JK = JK, drop_ratio = dropout_ratio, graph_pooling = graph_pooling, gnn_type = gnn_type)
    model.from_pretrained(MODEL_PATHS[model_name])
    model.graph_pred_linear = nn.Identity()
    model = model.to(device)
    model.eval()

    State.model = model
    State.model_name = model_name
    State.device = device
    State.batch_size = batch_size
    State.initialized = True


def process(smiles):
    data_list = []
    for i, x in enumerate(smiles):
        mol = AllChem.MolFromSmiles(x)
        data = mol_to_graph_data_obj_simple(mol)
        data.id = torch.tensor([i])
        data_list.append(data)
        
    data, slices = InMemoryDataset.collate(data_list)
    return data, slices

def encode(smiles: List[str]) -> np.ndarray:
    if not State.initialized:
        raise RuntimeError("Service is not setup, call 'setup' before 'encode'.")

    data, slices = process(smiles)
    dataset = InMemoryDataset()
    dataset._data = data
    dataset.slices = slices

    dataloader = DataLoader(dataset, State.batch_size)
    outputs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, f"Encoding with {State.model_name}"):
            batch = batch.to(State.device)
            ret = State.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            outputs.append(ret)
    outputs = torch.concat(outputs)
    outputs = outputs.cpu().numpy()
    return outputs
