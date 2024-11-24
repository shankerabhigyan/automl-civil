import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from nas import GeneticOptimizer
from src.xgb import CustomXGBoost
from src.anfis_sa import ANFISPredictor
import time
import yaml

def init_session_state():
    """Initialize session state variables"""
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = []
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'model_predictions' not in st.session_state:
        st.session_state.model_predictions = None
    if 'data' not in st.session_state:
        st.session_state.data = None

def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.title("Configuration")
        
        # Data Upload
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            # Target Column Selection
            target_column = st.selectbox(
                "Select Target Column",
                options=data.columns.tolist()
            )
            
            return data, target_column
    return None, None

def render_optimization_config():
    """Render optimization configuration section"""
    st.header("Optimization Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.slider(
            "Population Size",
            min_value=10,
            max_value=50,
            value=20,
            step=5
        )
        
        generations = st.slider(
            "Number of Generations",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )
        
    with col2:
        mutation_rate = st.slider(
            "Mutation Rate",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.1
        )
        
        elite_size = st.slider(
            "Elite Size",
            min_value=1,
            max_value=5,
            value=2,
            step=1
        )
    
    return {
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "elite_size": elite_size
    }

def plot_optimization_progress():
    """Plot optimization progress using Plotly"""
    if not st.session_state.optimization_history:
        return
    
    history_df = pd.DataFrame(st.session_state.optimization_history)
    
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
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_model_distribution():
    """Plot distribution of model types across generations"""
    if not st.session_state.optimization_history:
        return
    
    history_df = pd.DataFrame(st.session_state.optimization_history)
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
    
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance():
    """Plot feature importance if available"""
    if st.session_state.feature_importance is not None:
        importance_df = pd.DataFrame(st.session_state.feature_importance)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_results_tab():
    """Render the results tab"""
    if st.session_state.best_model is None:
        st.info("Run optimization to see results")
        return
    
    st.header("Best Model Details")
    
    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", st.session_state.best_model.model_type)
    with col2:
        st.metric("Fitness Score", f"{st.session_state.best_model.fitness:.4f}")
    with col3:
        st.metric("Parameters", len(st.session_state.best_model.params))
    
    # Model parameters
    st.subheader("Model Parameters")
    st.json(st.session_state.best_model.params)
    
    # Feature importance
    st.subheader("Feature Importance")
    plot_feature_importance()
    
    # Predictions vs Actual
    if st.session_state.model_predictions is not None:
        st.subheader("Predictions vs Actual")
        fig = px.scatter(
            x=st.session_state.data[st.session_state.target_column],
            y=st.session_state.model_predictions.flatten(),
            labels={'x': 'Actual', 'y': 'Predicted'},
            title='Predictions vs Actual Values'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_optimization_tab():
    """Render the optimization progress tab"""
    st.header("Optimization Progress")
    
    # Progress charts
    plot_optimization_progress()
    plot_model_distribution()
    
    # Detailed history table
    if st.session_state.optimization_history:
        st.subheader("Generation History")
        history_df = pd.DataFrame(st.session_state.optimization_history)
        st.dataframe(history_df, use_container_width=True)

def render_model_comparison_tab():
    """Render the model comparison tab"""
    st.header("Model Comparison")
    
    if not st.session_state.optimization_history:
        st.info("Run optimization to see model comparison")
        return
    
    # Compare different models' performance
    history_df = pd.DataFrame(st.session_state.optimization_history)
    
    # Box plot of fitness scores by model type
    fig = px.box(
        history_df,
        x='best_model_type',
        y='best_fitness',
        title='Fitness Score Distribution by Model Type',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.subheader("Performance Metrics by Model Type")
    metrics_df = history_df.groupby('best_model_type').agg({
        'best_fitness': ['mean', 'std', 'min', 'max']
    }).round(4)
    st.dataframe(metrics_df, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Neural Architecture Search Dashboard",
        page_icon="🧬",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("Neural Architecture Search Dashboard 🧬")
    
    # Render sidebar and get data
    data, target_column = render_sidebar()
    
    if data is not None and target_column is not None:
        st.session_state.target_column = target_column
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs([
            "Optimization", 
            "Results",
            "Model Comparison"
        ])
        
        with tab1:
            config = render_optimization_config()
            
            if st.button("Start Optimization", type="primary"):
                with st.spinner("Running optimization..."):
                    # Initialize and run optimizer
                    optimizer = GeneticOptimizer(**config)
                    best_model, history = optimizer.optimize(data, target_column)
                    
                    # Store results in session state
                    st.session_state.optimization_history = history
                    st.session_state.best_model = best_model
                    
                    # Get predictions if possible
                    try:
                        st.session_state.model_predictions = best_model.model.predict(data)
                        
                        # Get feature importance if available
                        if hasattr(best_model.model, 'plot_feature_importance'):
                            importance = best_model.model.get_feature_importance()
                            st.session_state.feature_importance = importance
                    except Exception as e:
                        st.error(f"Error getting predictions: {str(e)}")
                
                st.success("Optimization completed!")
            
            render_optimization_tab()
        
        with tab2:
            render_results_tab()
            
        with tab3:
            render_model_comparison_tab()
    else:
        st.info("Please upload a CSV file and select target column to begin")

if __name__ == "__main__":
    main()