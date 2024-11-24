import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

class SelfAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        return attended_features, attention_weights

class ANFISLayer(nn.Module):
    def __init__(self, num_features, num_rules=5):
        super().__init__()
        self.num_features = num_features
        self.num_rules = num_rules
        
        # Membership function parameters (Gaussian)
        self.mu = nn.Parameter(torch.randn(num_features, num_rules))
        self.sigma = nn.Parameter(torch.randn(num_features, num_rules))
        
        # Consequent parameters
        self.consequents = nn.Parameter(torch.randn(num_rules, num_features + 1))
        
    def membership_func(self, x):
        x = x.unsqueeze(2)
        return torch.exp(-(x - self.mu.unsqueeze(0))**2 / (2 * self.sigma.unsqueeze(0)**2))
    
    def forward(self, x):
        membership_values = self.membership_func(x)
        firing_strengths = torch.prod(membership_values, dim=1)
        normalized_firing_strengths = firing_strengths / (torch.sum(firing_strengths, dim=1, keepdim=True) + 1e-10)
        
        x_aug = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        consequent_outputs = torch.matmul(x_aug, self.consequents.t())
        
        output = torch.sum(normalized_firing_strengths * consequent_outputs, dim=1)
        return output.unsqueeze(1)

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], 
                 use_attention=True, attention_dim=64,
                 num_rules=5, dropout_rate=0.0):
        super().__init__()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(input_dim, attention_dim)
        
        self.anfis = ANFISLayer(input_dim, num_rules)
        self.ann = DynamicMLP(input_dim, hidden_dims, dropout_rate)
        self.combine = nn.Linear(2, 1)
        
    def forward(self, x):
        if self.use_attention:
            attended_features, attention_weights = self.attention(x)
        else:
            attended_features = x
            attention_weights = None
        
        anfis_out = self.anfis(attended_features)
        ann_out = self.ann(attended_features)
        
        combined = torch.cat([anfis_out, ann_out], dim=1)
        final_output = self.combine(combined)
        
        return final_output, attention_weights

class ANFISPredictor:
    def __init__(self, target_column, hidden_dims=[256, 128], 
                 use_attention=True, attention_dim=64,
                 num_rules=5, dropout_rate=0.0):
        self.target_column = target_column
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        self.attention_dim = attention_dim
        self.num_rules = num_rules
        self.dropout_rate = dropout_rate
        
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        
    def preprocess_data(self, data):
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column].values.reshape(-1, 1)
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled
        
    def fit(self, data, epochs=100, batch_size=32, learning_rate=0.001):
        X_scaled, y_scaled = self.preprocess_data(data)
        
        # Initialize model
        self.model = HybridModel(
            input_dim=X_scaled.shape[1],
            hidden_dims=self.hidden_dims,
            use_attention=self.use_attention,
            attention_dim=self.attention_dim,
            num_rules=self.num_rules,
            dropout_rate=self.dropout_rate
        )
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
                
    def predict(self, data):
        X = data.drop(columns=[self.target_column])
        X_scaled = self.scaler_X.transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
        
        return self.scaler_y.inverse_transform(predictions.numpy())
    
    def get_fitness_score(self, data):
        pred = self.predict(data)
        true = data[self.target_column].values
        score = np.mean((true - pred)**2)   # MSE
        rmse = np.sqrt(score)               # RMSE
        # return fitness score
        return 1/(1+rmse)

# Example usage:
if __name__ == "__main__":
    # Load data
    path = "Concrete_Data_Yeh.csv"
    data = pd.read_csv(path)
    target = 'csMPa'
    
    # Initialize and train model with custom parameters
    model = ANFISPredictor(
        target_column=target,
        hidden_dims=[256, 128, 64],  # Custom architecture
        use_attention=True,          # Enable self-attention
        attention_dim=32,            # Custom attention dimension
        num_rules=7,                 # Custom number of ANFIS rules
        dropout_rate=0.2            # Add dropout for regularization
    )
    
    # Train the model
    model.fit(data, epochs=100, batch_size=32, learning_rate=0.001)
    
    # Make predictions
    predictions = model.predict(data)
    print("Predictions shape:", predictions.shape)