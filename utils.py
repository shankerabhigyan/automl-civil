import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import streamlit as st

def create_optimization_progress_plot(history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive plot showing optimization progress
    
    Args:
        history: List of dictionaries containing optimization history
        
    Returns:
        Plotly figure object
    """
    history_df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    # Add traces for best and average fitness
    fig.add_trace(go.Scatter(
        x=history_df['generation'],
        y=history_df['best_fitness'],
        name='Best Fitness',
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df['generation'],
        y=history_df['avg_fitness'],
        name='Average Fitness',
        line=dict(color='#3498db', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Optimization Progress',
        xaxis_title='Generation',
        yaxis_title='Fitness Score',
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_model_distribution_plot(history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a plot showing distribution of model types across generations
    
    Args:
        history: List of dictionaries containing optimization history
        
    Returns:
        Plotly figure object
    """
    history_df = pd.DataFrame(history)
    model_counts = pd.DataFrame({
        'generation': history_df['generation'],
        'model_type': history_df['best_model_type']
    })
    
    fig = px.histogram(
        model_counts,
        x='generation',
        color='model_type',
        title='Best Model Type Distribution Across Generations',
        template='plotly_white'
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_feature_importance_plot(importance_df: pd.DataFrame) -> go.Figure:
    """
    Create a feature importance plot
    
    Args:
        importance_df: DataFrame containing feature importance scores
        
    Returns:
        Plotly figure object
    """
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance',
        template='plotly_white'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_predictions_plot(actual: np.ndarray, predicted: np.ndarray, 
                          target_name: str) -> go.Figure:
    """
    Create a scatter plot of predicted vs actual values
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        target_name: Name of target variable
        
    Returns:
        Plotly figure object
    """
    fig = px.scatter(
        x=actual,
        y=predicted.flatten(),
        labels={'x': f'Actual {target_name}', 'y': f'Predicted {target_name}'},
        title='Predictions vs Actual Values'
    )
    
    # Add diagonal line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_model_comparison_plot(history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a box plot comparing model performance for all model types
    
    Args:
        history: List of dictionaries containing optimization history
        
    Returns:
        Plotly figure object
    """
    history_df = pd.DataFrame(history)
    
    # Ensure all model types are represented
    all_model_types = ['anfis', 'xgboost', 'logistic', 'svm']
    
    # Create a complete dataset with all model types
    complete_data = []
    for model_type in all_model_types:
        # Get existing data for this model type
        model_data = history_df[history_df['best_model_type'] == model_type]
        
        if len(model_data) == 0:
            # If no data exists for this model type, add a placeholder with NaN
            complete_data.append({
                'best_model_type': model_type,
                'best_fitness': np.nan,
                'generation': 0
            })
    
    # Combine existing data with placeholders
    complete_df = pd.concat([
        history_df,
        pd.DataFrame(complete_data)
    ]).fillna(0)  # Replace NaN with 0 for visualization
    
    fig = px.box(
        complete_df,
        x='best_model_type',
        y='best_fitness',
        title='Fitness Score Distribution by Model Type',
        template='plotly_white'
    )
    
    fig.update_layout(
        xaxis_title='Model Type',
        yaxis_title='Fitness Score',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_model_evolution_plot(history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a plot showing how each model type's performance evolved over generations
    
    Args:
        history: List of dictionaries containing optimization history
        
    Returns:
        Plotly figure object
    """
    history_df = pd.DataFrame(history)
    
    # Get unique model types
    model_types = history_df['best_model_type'].unique()
    
    fig = go.Figure()
    
    for model_type in model_types:
        model_data = history_df[history_df['best_model_type'] == model_type]
        
        fig.add_trace(go.Scatter(
            x=model_data['generation'],
            y=model_data['best_fitness'],
            name=model_type,
            mode='lines+markers',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Model Performance Evolution',
        xaxis_title='Generation',
        yaxis_title='Fitness Score',
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def calculate_model_statistics(history: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for each model type
    
    Args:
        history: List of dictionaries containing optimization history
        
    Returns:
        DataFrame containing model statistics
    """
    history_df = pd.DataFrame(history)
    all_model_types = ['anfis', 'xgboost', 'logistic', 'svm']
    
    stats = []
    for model_type in all_model_types:
        model_data = history_df[history_df['best_model_type'] == model_type]
        
        if len(model_data) > 0:
            stats.append({
                'Model Type': model_type,
                'Times Selected': len(model_data),
                'Selection Rate (%)': (len(model_data) / len(history_df)) * 100,
                'Average Fitness': model_data['best_fitness'].mean(),
                'Best Fitness': model_data['best_fitness'].max(),
                'Worst Fitness': model_data['best_fitness'].min(),
                'Fitness Std Dev': model_data['best_fitness'].std(),
                'First Selected': model_data['generation'].min(),
                'Last Selected': model_data['generation'].max()
            })
        else:
            stats.append({
                'Model Type': model_type,
                'Times Selected': 0,
                'Selection Rate (%)': 0,
                'Average Fitness': 0,
                'Best Fitness': 0,
                'Worst Fitness': 0,
                'Fitness Std Dev': 0,
                'First Selected': np.nan,
                'Last Selected': np.nan
            })
    
    return pd.DataFrame(stats)