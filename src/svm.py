import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import List, Tuple, Optional, Dict
from torch.nn.functional import relu

class KernelLayer(nn.Module):
    def __init__(self, kernel_type: str = 'rbf', kernel_params: Dict = None):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {}
        
        if kernel_type == 'rbf':
            self.gamma = nn.Parameter(torch.tensor(
                self.kernel_params.get('gamma', 1.0)
            ))
        elif kernel_type == 'poly':
            self.degree = self.kernel_params.get('degree', 3)
            self.coef0 = nn.Parameter(torch.tensor(
                self.kernel_params.get('coef0', 1.0)
            ))
    
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = x1 if x2 is None else x2
        
        if self.kernel_type == 'linear':
            return torch.mm(x1, x2.t())
        
        elif self.kernel_type == 'rbf':
            # Compute RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
            dists = torch.cdist(x1, x2, p=2)
            return torch.exp(-self.gamma * (dists ** 2))
        
        elif self.kernel_type == 'poly':
            # Compute polynomial kernel: K(x,y) = (x·y + coef0)^degree
            linear = torch.mm(x1, x2.t())
            return (linear + self.coef0) ** self.degree
        
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

class FeatureSelector(nn.Module):
    def __init__(self, input_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.feature_weights = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_probs = torch.sigmoid(self.feature_weights / self.temperature)
        mask = (feature_probs > 0.5).float()
        selected_features = x * mask.unsqueeze(0)
        return selected_features, mask

class SVMModel(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 kernel_type: str = 'rbf',
                 kernel_params: Dict = None,
                 use_feature_selection: bool = True,
                 temperature: float = 1.0,
                 C: float = 1.0):
        super().__init__()
        
        self.C = C
        self.use_feature_selection = use_feature_selection
        
        if use_feature_selection:
            self.feature_selector = FeatureSelector(input_dim, temperature)
            
        self.kernel = KernelLayer(kernel_type, kernel_params)
        self.alpha = nn.Parameter(torch.zeros(1))  # Will be expanded during training
        self.bias = nn.Parameter(torch.zeros(1))
        self.support_vectors = None
        
    def get_kernel_matrix(self, x: torch.Tensor, support_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        if support_vectors is None:
            return self.kernel(x)
        return self.kernel(x, support_vectors)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feature_mask = None
        if self.use_feature_selection:
            x, feature_mask = self.feature_selector(x)
        
        kernel_matrix = self.get_kernel_matrix(x, self.support_vectors)
        outputs = torch.mm(kernel_matrix, self.alpha) + self.bias
        return torch.sigmoid(outputs), feature_mask
    
    def hinge_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Modified hinge loss with sigmoid outputs
        margin = targets * (2 * outputs - 1)
        return torch.mean(relu(1 - margin))
    
    def regularization_loss(self) -> torch.Tensor:
        return 0.5 * torch.sum(self.alpha ** 2)

class SVMNAS:
    def __init__(self,
                 target_column: str,
                 kernel_type: str = 'rbf',
                 kernel_params: Dict = None,
                 use_feature_selection: bool = True,
                 temperature: float = 1.0,
                 C: float = 1.0):
        
        self.target_column = target_column
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {}
        self.use_feature_selection = use_feature_selection
        self.temperature = temperature
        self.C = C
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.support_vectors = None
        
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
            support_vector_ratio: float = 0.1) -> List[float]:
        
        X_scaled, y = self.preprocess_data(data)
        
        # Select initial support vectors (random subset of training data)
        n_support = max(int(X_scaled.shape[0] * support_vector_ratio), 1)
        support_indices = np.random.choice(X_scaled.shape[0], n_support, replace=False)
        
        
        # Initialize model
        self.model = SVMModel(
            input_dim=X_scaled.shape[1],
            kernel_type=self.kernel_type,
            kernel_params=self.kernel_params,
            use_feature_selection=self.use_feature_selection,
            temperature=self.temperature,
            C=self.C
        )
        self.model.support_vectors = torch.FloatTensor(X_scaled[support_indices])
        
        # Initialize alpha parameter
        self.model.alpha = nn.Parameter(torch.zeros(n_support, 1))
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        training_history = []
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                
                # Calculate losses
                hinge_loss = self.model.hinge_loss(outputs, batch_y)
                reg_loss = self.model.regularization_loss()
                loss = hinge_loss + (1 / (2 * self.C)) * reg_loss
                
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
            predictions, _ = self.model(X_tensor)
        
        return (predictions.numpy() > 0.5).astype(int)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        X = data[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
        
        return predictions.numpy()
    
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
    
    def get_support_vectors(self) -> np.ndarray:
        """Return the support vectors used by the model"""
        return self.model.support_vectors.numpy()
    
    def get_fitness_score(self, data: pd.DataFrame) -> float:
        """
        Calculate fitness score for genetic evolution
        Returns a score between 0 and 1, where higher is better
        """
        y_pred = self.predict(data)
        y_true = data[self.target_column].values
        
        # Calculate multiple metrics
        accuracy = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, self.predict_proba(data))
        except:
            auc = 0.5
        
        # Feature sparsity bonus
        sparsity_bonus = 0
        if self.use_feature_selection:
            feature_mask = self.get_feature_importance() > 0.5
            sparsity_bonus = 0.1 * (1 - feature_mask.mean())
        
        # Support vector sparsity bonus
        sv_ratio = len(self.get_support_vectors()) / len(data)
        sv_bonus = 0.1 * (1 - sv_ratio)
        
        # Combine metrics
        fitness = 0.4 * accuracy + 0.3 * auc + 0.15 * sparsity_bonus + 0.15 * sv_bonus
        return fitness

# Example usage:
if __name__ == "__main__":
    # Load data (example with binary classification dataset)
    path = "classification_data.csv"
    data = pd.read_csv(path)
    target = 'target'
    
    # Initialize and train model with custom parameters
    model = SVMNAS(
        target_column=target,
        kernel_type='rbf',           # Use RBF kernel
        kernel_params={'gamma': 1.0},
        use_feature_selection=True,  # Enable feature selection
        temperature=1.0,            # Temperature for feature selection
        C=1.0                       # Regularization parameter
    )
    
    # Train the model
    history = model.fit(
        data,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        support_vector_ratio=0.1
    )
    
    # Make predictions
    predictions = model.predict(data)
    print("Predictions shape:", predictions.shape)
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Get support vectors
    support_vectors = model.get_support_vectors()
    print(f"\nNumber of support vectors: {len(support_vectors)}")
    
    # Calculate fitness score
    fitness = model.get_fitness_score(data)
    print(f"\nFitness Score: {fitness:.4f}")