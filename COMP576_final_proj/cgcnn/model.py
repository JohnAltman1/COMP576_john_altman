from __future__ import print_function, division

import torch
import torch.nn as nn

NI_ID = 28
CU_ID = 29

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=192, n_h=3,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
    
        self.conv_to_fc = nn.Linear(atom_fea_len * 2, h_fea_len)
  

        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, atom_types):
        """
        Forward pass

        New Parameter:
        atom_types: torch.LongTensor shape (N,)
             Vector containing atomic numbers for every atom in the batch.
             Must use 28 for Ni and 29 for Cu.
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Pass atom_types to pooling
        crys_fea = self.pooling(atom_fea, crystal_atom_idx, atom_types)
        
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx, atom_types):
        """
        Pooling the atom features to crystal features by element type.
        Only pools Ni and Cu.
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]

        # Only pool Ni and Cu
        target_elements = [NI_ID, CU_ID]
        
        batch_crystal_features = []

        for i, idx_map in enumerate(crystal_atom_idx):
            local_features = atom_fea[idx_map]
            local_types = atom_types[idx_map]
            
            element_pools = []
            
            for elem_id in target_elements:
                # Identify indices of the specific element
                mask = (local_types == elem_id).nonzero(as_tuple=False).squeeze()
                
                if mask.numel() > 0:
                    # Handle single atom vs multiple atoms
                    if mask.dim() == 0:
                        mapped_feats = local_features[mask].unsqueeze(0)
                    else:
                        mapped_feats = local_features[mask]
                    
                    # Mean pooling for this element
                    pooled = torch.mean(mapped_feats, dim=0, keepdim=False)
                else:
                    # This prevents crashes on pure Ni or pure Cu structures
                    pooled = torch.zeros(self.convs[0].atom_fea_len, 
                                         device=atom_fea.device, 
                                         dtype=atom_fea.dtype)
                
                element_pools.append(pooled)
            
            # Concatenate [Ni_vec, Cu_vec]
            crystal_vec = torch.cat(element_pools, dim=0)
            batch_crystal_features.append(crystal_vec.unsqueeze(0))

        return torch.cat(batch_crystal_features, dim=0)