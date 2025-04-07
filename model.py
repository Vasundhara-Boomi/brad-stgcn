import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- GCN Layer with Shape Debugging --------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, adjacency):
        # Print shapes for debugging
        print(f"GCNLayer Input shapes - x: {x.shape}, adjacency: {adjacency.shape}")
        
        # Ensure x is [B, T, V, C]
        if len(x.shape) == 3:  # [T, V, C]
            x = x.unsqueeze(0)  # Add batch dim
        
        # Ensure adjacency is compatible
        if len(adjacency.shape) == 2:  # [V, V]
            adjacency = adjacency.unsqueeze(0)  # Add batch dim
        
        if len(adjacency.shape) == 3:  # [B, V, V]
            # Expand to match time dimension of x
            adjacency = adjacency.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        
        # Verify shapes after adjustments
        print(f"After adjustment - x: {x.shape}, adjacency: {adjacency.shape}")
        
        # Get batch, time, nodes, features dimensions
        B, T, V, C = x.shape
        
        # For matrix multiplication, we need to process each batch and time step separately
        x_out = []
        for b in range(B):
            time_steps = []
            for t in range(T):
                # Get adjacency for this batch and time step
                adj_bt = adjacency[b, t]  # [V, V]
                
                # Get features for this batch and time step
                x_bt = x[b, t]  # [V, C]
                
                # Normalize adjacency (add self-loops if needed)
                if adj_bt.shape[0] == adj_bt.shape[1]:  # Only if square matrix
                    degree = adj_bt.sum(dim=1, keepdim=True)  # [V, 1]
                    degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
                    adj_norm = degree_inv_sqrt * adj_bt * degree_inv_sqrt.transpose(0, 1)
                    
                    # Apply convolution: adj_norm [V, V] × x_bt [V, C]
                    x_conv = torch.matmul(adj_norm, x_bt)  # Should be [V, C]
                    time_steps.append(x_conv)
                else:
                    print(f"WARNING: Non-square adjacency matrix: {adj_bt.shape}")
                    # Just pass through if adjacency isn't square
                    time_steps.append(x_bt)
            
            # Stack time steps
            x_out.append(torch.stack(time_steps))
        
        # Stack batches
        x = torch.stack(x_out)  # [B, T, V, C]
        
        # Apply linear transformation
        # Linear layer expects last dimension to match in_features
        if C != self.in_features:
            print(f"WARNING: Input features mismatch. Expected {self.in_features}, got {C}.")
            # Handle mismatch by creating a temporary adapter
            adapter = nn.Linear(C, self.in_features).to(x.device)
            x = adapter(x.view(-1, C)).view(B, T, V, self.in_features)
            C = self.in_features
        
        # Reshape for linear layer
        x_linear = self.linear(x.view(-1, C)).view(B, T, V, self.out_features)
        
        # Batch normalization expects [N, C, *]
        x_bn = x_linear.permute(0, 3, 1, 2)  # [B, C, T, V]
        shape = x_bn.shape
        x_bn = x_bn.reshape(B, self.out_features, -1)  # [B, C, T*V]
        x_bn = self.bn(x_bn)
        x_bn = x_bn.reshape(shape)  # [B, C, T, V]
        x_bn = x_bn.permute(0, 2, 3, 1)  # [B, T, V, C]
        
        # Apply ReLU
        x_out = F.relu(x_bn)
        
        print(f"GCNLayer Output shape: {x_out.shape}")
        return x_out

# -------------------- ST-GCN Model with Shape Handling --------------------
class STGCN(nn.Module):
    def __init__(self, in_features, gcn_hidden, tcn_hidden, num_classes, num_nodes, kernel_size=3, dropout_rate=0.3):
        super(STGCN, self).__init__()
        
        # Save parameters
        self.in_features = in_features
        self.gcn_hidden = gcn_hidden
        self.tcn_hidden = tcn_hidden
        self.num_nodes = num_nodes
        
        # Spatial GCN Layers
        self.gcn1 = GCNLayer(in_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)

        # Temporal Convolution with shape adaptation
        self.tcn = nn.Conv1d(gcn_hidden * num_nodes, tcn_hidden, kernel_size, padding=kernel_size//2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(tcn_hidden, tcn_hidden // 2)
        self.fc2 = nn.Linear(tcn_hidden // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adjacency):
        # Print input shapes for debugging
        print(f"STGCN Input shapes - x: {x.shape}, adjacency: {adjacency.shape}")
        
        # Validate and adapt input shapes
        if len(x.shape) == 3:  # [T, V, C]
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Check if we need to adapt features dimension
        B, T, V, C = x.shape
        if C != self.in_features:
            print(f"WARNING: Input features mismatch. Expected {self.in_features}, got {C}.")
            # Two options: adapt or truncate/pad
            if C > self.in_features:
                # Truncate
                x = x[..., :self.in_features]
            else:
                # Pad with zeros
                padding = torch.zeros(B, T, V, self.in_features - C, device=x.device)
                x = torch.cat([x, padding], dim=-1)
        
        # Spatial graph convolution
        x1 = self.gcn1(x, adjacency)
        x2 = self.gcn2(x1, adjacency)
        x = x1 + x2  # Residual connection
        
        # Print shape after GCN
        print(f"Shape after GCN: {x.shape}")
        
        # Reshape for TCN: [B, T, V, C] -> [B, V*C, T]
        B, T, V, C = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, V, C, T]
        x = x.reshape(B, V * C, T)  # [B, V*C, T]
        
        # Print shape before TCN
        print(f"Shape before TCN: {x.shape}")
        
        # Apply temporal convolution
        x = self.tcn(x)
        
        # Print shape after TCN
        print(f"Shape after TCN: {x.shape}")
        
        # Global average pooling over time
        x = x.mean(dim=-1)  # [B, tcn_hidden]
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Print final output shape
        print(f"Final output shape: {x.shape}")
        return x