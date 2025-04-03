import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model parameters
in_features = 6  
num_nodes = 21
num_classes = 4  
gcn_hidden = 64
lstm_hidden = 64
dropout_rate = 0.4

# GCN Layer definition
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, adjacency):
        degree = adjacency.sum(dim=-1, keepdim=True)
        degree_inv_sqrt = degree.pow(-0.5)
        adjacency = degree_inv_sqrt * adjacency * degree_inv_sqrt

        x = torch.matmul(adjacency, x)
        x = self.linear(x)
        x = self.bn(x.view(-1, x.shape[-1])).view(x.shape)
        x = F.relu(x)
        return x

# STGCN Model
class STGCN(nn.Module):
    def __init__(self, in_features, gcn_hidden, lstm_hidden, num_classes, num_nodes, dropout_rate=0.3):
        super(STGCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        
        self.lstm = nn.LSTM(
            gcn_hidden * num_nodes, 
            lstm_hidden, 
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if 2 > 1 else 0
        )
        
        self.fc1 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.fc2 = nn.Linear(lstm_hidden // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adjacency):
        # First GCN layer
        x1 = self.gcn1(x, adjacency)
        
        # Second GCN layer with residual connection
        x2 = self.gcn2(x1, adjacency)
        x = x1 + x2  # Residual connection
        
        x = self.dropout(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use last timestep
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# EMA for parameter smoothing
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]

# Linear warmup scheduler
class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, initial_lr=1e-4, max_lr=0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch / self.warmup_epochs)
        else:
            # Keep constant learning rate after warmup
            self.current_lr = self.max_lr
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def get_last_lr(self):
        return [self.current_lr]

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Initialize model, optimizer, and schedulers
model = STGCN(in_features, gcn_hidden, lstm_hidden, num_classes, num_nodes, dropout_rate).to(device)
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)

scheduler = LinearWarmupScheduler(
    optimizer,
    warmup_epochs=10,
    total_epochs=200,
    initial_lr=1e-4,
    max_lr=0.001
)

ema = EMA(model, decay=0.99)
ema.register()




# Function to load the model
def load_model(model_path):
    checkpoint = torch.load(model_path)
    
    # Create model with saved parameters
    model = STGCN(
        in_features=checkpoint['model_params']['in_features'],
        gcn_hidden=checkpoint['model_params']['gcn_hidden'],
        lstm_hidden=checkpoint['model_params']['lstm_hidden'],
        num_classes=checkpoint['model_params']['num_classes'],
        num_nodes=checkpoint['model_params']['num_nodes'],
        dropout_rate=checkpoint['model_params']['dropout_rate']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint
