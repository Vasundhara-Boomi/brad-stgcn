import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(GCNLayer, self).__init__()
        self.num_nodes = num_nodes
        self.out_features = out_features

        # Correct weight initialization with the right dimensions
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        # Batch normalization with correct number of features
        self.bn = nn.BatchNorm1d(out_features)

    def normalize_adjacency(self, adjacency):
        """Normalize adjacency matrix."""
        adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device)
        degree = adjacency.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        degree_inv_sqrt_diag = torch.diag(degree_inv_sqrt)
        return torch.matmul(torch.matmul(degree_inv_sqrt_diag, adjacency), degree_inv_sqrt_diag)

    def forward(self, x, adjacency):
        batch_size, num_nodes, timesteps, in_features = x.shape
        
        # Normalize adjacency matrix
        adjacency_norm = self.normalize_adjacency(adjacency)

        # Reshape input for processing
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.view(-1, num_nodes, in_features)

        # Create output tensor
        x_transformed = torch.zeros(x_reshaped.shape[0], num_nodes, self.out_features, 
                                    device=x.device, dtype=x.dtype)

        # Process each timestep
        for i in range(x_reshaped.shape[0]):
            # Perform feature transformation
            node_features = x_reshaped[i]  # [num_nodes, in_features]
            transformed = torch.matmul(node_features, self.weight)  # [num_nodes, out_features]
            x_transformed[i] = transformed

        # Reshape back to original structure
        x_transformed = x_transformed.view(batch_size, timesteps, num_nodes, -1)
        x_transformed = x_transformed.permute(0, 2, 1, 3).contiguous()

        # Graph convolution using adjacency
        x_conv = torch.zeros_like(x_transformed)
        for b in range(batch_size):
            for t in range(timesteps):
                x_conv[b, :, t, :] = torch.matmul(adjacency_norm, x_transformed[b, :, t, :])

        # Batch normalization - reshape to apply across node features
        x_bn = x_conv.view(-1, self.out_features)
        x_bn = self.bn(x_bn)
        x_bn = x_bn.view(batch_size, num_nodes, timesteps, -1)

        # Activation
        x_out = F.relu(x_bn)

        return x_out
    
class STGCN(nn.Module):
    def __init__(self, in_features, gcn_hidden, lstm_hidden, num_classes, num_nodes, dropout_rate=0.3):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes

        # Graph Convolutional Layers
        self.gcn1 = GCNLayer(in_features=9, out_features=gcn_hidden, num_nodes=num_nodes)
        self.gcn2 = GCNLayer(in_features=gcn_hidden, out_features=gcn_hidden, num_nodes=num_nodes)

        # LSTM Layer
        self.lstm = nn.LSTM(
            gcn_hidden * num_nodes,
            lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if 2 > 1 else 0
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.fc2 = nn.Linear(lstm_hidden // 2, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adjacency):
        print("\n[STGCN Forward] Initial Input Shape:", x.shape)

        # First GCN layer
        x1 = self.gcn1(x, adjacency)
        print("[STGCN Forward] Output of GCN1 Shape:", x1.shape)

        # Second GCN layer with residual connection
        x2 = self.gcn2(x1, adjacency)
        print("[STGCN Forward] Output of GCN2 Shape:", x2.shape)

        x = x1 + x2  # Residual connection
        print("[STGCN Forward] Residual Connection Shape:", x.shape)

        # Reshape for LSTM
        batch_size, num_nodes, timesteps, features = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, timesteps, -1)
        print("[STGCN Forward] LSTM Input Shape:", x.shape)

        # Dropout
        x = self.dropout(x)

        # LSTM
        x, _ = self.lstm(x)
        print("[STGCN Forward] LSTM Output Shape:", x.shape)

        # Use last timestep
        x = x[:, -1, :]
        print("[STGCN Forward] After Selecting Last Timestep:", x.shape)

        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        print("[STGCN Forward] After FC1:", x.shape)

        x = self.dropout(x)
        x = self.fc2(x)
        print("[STGCN Forward] Final Output Shape:", x.shape)

        return x
