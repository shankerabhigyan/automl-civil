import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
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
        x = x.unsqueeze(2)  # Add dimension for rules
        return torch.exp(-(x - self.mu.unsqueeze(0))**2 / (2 * self.sigma.unsqueeze(0)**2))
    
    def forward(self, x):
        # Calculate membership degrees
        membership_values = self.membership_func(x)
        
        # Calculate firing strengths
        firing_strengths = torch.prod(membership_values, dim=1)
        normalized_firing_strengths = firing_strengths / (torch.sum(firing_strengths, dim=1, keepdim=True) + 1e-10)
        
        # Calculate consequent outputs
    
        x_aug = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        consequent_outputs = torch.matmul(x_aug, self.consequents.t())
        
        # Calculate final output
        output = torch.sum(normalized_firing_strengths * consequent_outputs, dim=1)
        return output.unsqueeze(1)

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attention = SelfAttention(input_dim)
        self.anfis = ANFISLayer(input_dim)
        
        self.ann = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.combine = nn.Linear(2, 1)
        
    def forward(self, x):
        # Apply self-attention
        attended_features, attention_weights = self.attention(x)
        
        # ANFIS path
        anfis_out = self.anfis(attended_features)
        
        # ANN path
        ann_out = self.ann(attended_features)
        
        # Combine predictions
        combined = torch.cat([anfis_out, ann_out], dim=1)
        final_output = self.combine(combined)
        
        return final_output, attention_weights

class FeatureSelector:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.importance_scores = None
        
    def fit(self, X, y, epochs=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridModel(self.input_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs, attention_weights = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        # Store feature importance based on attention weights
        _, attention_weights = model(X_tensor)
        self.importance_scores = attention_weights.mean(dim=0).cpu().detach().numpy()
        
    def select_features(self, threshold=0.1):
        return self.importance_scores >= threshold

class AutoMLConcrete:
    def __init__(self, target_column):
        self.target_column = target_column
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_selector = None
        self.model = None
        self.selected_features = None
        
    def preprocess_data(self, data):
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column].values.reshape(-1, 1)
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled
        
    def fit(self, data, epochs=100):
        X_scaled, y_scaled = self.preprocess_data(data)
        
        # Feature selection using NAS
        self.feature_selector = FeatureSelector(X_scaled.shape[1])
        self.feature_selector.fit(X_scaled, y_scaled)
        self.selected_features = self.feature_selector.select_features()
        
        # Train final model using selected features
        X_selected = X_scaled[:, self.selected_features]
        self.model = HybridModel(X_selected.shape[1])
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_selected)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Training loop
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs, _ = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                
    def predict(self, data):
        X = data.drop(columns=[self.target_column])
        X_scaled = self.scaler_X.transform(X)
        X_selected = X_scaled[:, self.selected_features]
        
        X_tensor = torch.FloatTensor(X_selected)
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
        
        # Inverse transform predictions
        return self.scaler_y.inverse_transform(predictions.numpy())

# Example usage:
if __name__ == "__main__":
    # # Generate synthetic data
    # np.random.seed(42)
    # n_samples = 1000
    # n_features = 10
    
    # X = np.random.randn(n_samples, n_features)
    # y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(n_samples)*0.1
    
    # # Create DataFrame
    # data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    # data['target'] = y
    
    path = "../data/concrete/Concrete_Data_Yeh.csv"
    data = pd.read_csv(path)
    target = 'csMPa'
    
    # Initialize and train model
    automl = AutoMLConcrete(target_column=target)
    automl.fit(data)
    
    # Make predictions
    predictions = automl.predict(data)
    print("Predictions shape:", predictions.shape)