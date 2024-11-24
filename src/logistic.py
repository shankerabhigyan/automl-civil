import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import List, Tuple, Optional

class SelfAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        return attended_features, attention_weights

class LogisticLayer(nn.Module):
    def __init__(self, input_dim: int, regularization: str = 'l2'):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.regularization = regularization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))
    
    def get_regularization_loss(self) -> torch.Tensor:
        if self.regularization == 'l1':
            return torch.sum(torch.abs(self.linear.weight))
        elif self.regularization == 'l2':
            return torch.sum(self.linear.weight ** 2)
        else:
            return torch.tensor(0.0)

class FeatureSelector(nn.Module):
    def __init__(self, input_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.feature_weights = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_probs = torch.sigmoid(self.feature_weights / self.temperature)
        # Using straight-through estimator for discrete selection
        mask = (feature_probs > 0.5).float()
        selected_features = x * mask.unsqueeze(0)
        return selected_features, mask

class HybridLogisticModel(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 use_attention: bool = True,
                 attention_dim: int = 64,
                 use_feature_selection: bool = True,
                 regularization: str = 'l2',
                 temperature: float = 1.0):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_feature_selection = use_feature_selection
        
        if use_feature_selection:
            self.feature_selector = FeatureSelector(input_dim, temperature)
            
        if use_attention:
            self.attention = SelfAttention(input_dim, attention_dim)
            
        self.logistic = LogisticLayer(input_dim, regularization)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        feature_mask = None
        attention_weights = None
        
        if self.use_feature_selection:
            x, feature_mask = self.feature_selector(x)
            
        if self.use_attention:
            x, attention_weights = self.attention(x)
            
        output = self.logistic(x)
        return output, attention_weights, feature_mask

class LogisticNAS:
    def __init__(self,
                 target_column: str,
                 use_attention: bool = True,
                 attention_dim: int = 64,
                 use_feature_selection: bool = True,
                 regularization: str = 'l2',
                 temperature: float = 1.0):
        
        self.target_column = target_column
        self.use_attention = use_attention
        self.attention_dim = attention_dim
        self.use_feature_selection = use_feature_selection
        self.regularization = regularization
        self.temperature = temperature
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.feature_names = [col for col in data.columns if col != self.target_column]
        X = data[self.feature_names].values
        y = data[self.target_column].values
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
        
    def fit(self, 
            data: pd.DataFrame,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            reg_strength: float = 0.01) -> List[float]:
        
        X_scaled, y = self.preprocess_data(data)
        
        # Initialize model
        self.model = HybridLogisticModel(
            input_dim=X_scaled.shape[1],
            use_attention=self.use_attention,
            attention_dim=self.attention_dim,
            use_feature_selection=self.use_feature_selection,
            regularization=self.regularization,
            temperature=self.temperature
        )
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        training_history = []
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs, _, feature_mask = self.model(batch_X)
                
                # Calculate main loss
                loss = criterion(outputs, batch_y)
                
                # Add regularization loss
                if self.regularization in ['l1', 'l2']:
                    reg_loss = self.model.logistic.get_regularization_loss()
                    loss += reg_strength * reg_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            training_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
        
        return training_history
                
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = data[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            predictions, _, feature_mask = self.model(X_tensor)
        
        return predictions.numpy()
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict(data)
    
    def get_feature_importance(self) -> pd.Series:
        if not self.use_feature_selection:
            return None
        
        with torch.no_grad():
            feature_probs = torch.sigmoid(self.model.feature_selector.feature_weights)
        
        return pd.Series(
            feature_probs.numpy(),
            index=self.feature_names,
            name='Feature Importance'
        ).sort_values(ascending=False)
    
    def get_fitness_score(self, data: pd.DataFrame) -> float:
        """
        Calculate fitness score for genetic evolution
        Returns a score between 0 and 1, where higher is better
        """
        y_pred = self.predict(data)
        y_true = data[self.target_column].values
        
        # Calculate multiple metrics
        accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = 0.5  # Default value if AUC cannot be calculated
            
        # Feature sparsity bonus (if feature selection is used)
        sparsity_bonus = 0
        if self.use_feature_selection:
            feature_mask = self.get_feature_importance() > 0.5
            sparsity_bonus = 0.1 * (1 - feature_mask.mean())
            
        # Combine metrics into final fitness score
        fitness = 0.4 * accuracy + 0.4 * auc + 0.2 * sparsity_bonus
        return fitness

# Example usage:
if __name__ == "__main__":
    # Load data (example with binary classification dataset)
    path = "classification_data.csv"
    data = pd.read_csv(path)
    target = 'target'
    
    # Initialize and train model with custom parameters
    model = LogisticNAS(
        target_column=target,
        use_attention=True,          # Enable self-attention
        attention_dim=32,            # Custom attention dimension
        use_feature_selection=True,  # Enable feature selection
        regularization='l2',         # Use L2 regularization
        temperature=1.0             # Temperature for feature selection
    )
    
    # Train the model
    history = model.fit(
        data,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        reg_strength=0.01
    )
    
    # Make predictions
    predictions = model.predict(data)
    print("Predictions shape:", predictions.shape)
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Calculate fitness score
    fitness = model.get_fitness_score(data)
    print(f"\nFitness Score: {fitness:.4f}")