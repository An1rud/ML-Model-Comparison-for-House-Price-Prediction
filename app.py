import streamlit as st
import numpy as np
import pandas as pd
import time
import psutil
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# Page configuration
st.set_page_config(
    page_title="ML Model Comparison for House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏠 ML Model Comparison for House Price Prediction")
st.markdown("""
This application provides a comprehensive comparison of different machine learning algorithms 
for house price prediction using the California Housing dataset.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    ["🏠 Home", "📊 Dataset Overview", "🔧 Model Training", "📈 Results & Comparison", "🧮 Complexity Analysis"]
)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'complexity_df' not in st.session_state:
    st.session_state.complexity_df = None

def load_data():
    """Load and preprocess the California Housing dataset"""
    try:
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        return X, y, data.DESCR
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None

def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split and scale the data"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_models(n_estimators=100, random_state=42):
    """Initialize the ML models with configurable parameters"""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state),
        'SVR': SVR(kernel='rbf')
    }
    return models

def train_and_evaluate_models(models, X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate all models"""
    results = {
        'Model': [],
        'R2 Score': [],
        'MSE': [],
        'MAE': [],
        'Training Time (s)': [],
        'Prediction Time (s)': [],
        'Memory Usage (MB)': []
    }
    
    model_objects = {}
    predictions = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        
        # Measure memory usage
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # in MB
        
        # Training time
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = model.predict(X_test_scaled)
        prediction_time = time.time() - start_time
        
        # Memory after training
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_usage = mem_after - mem_before
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store results
        results['Model'].append(name)
        results['R2 Score'].append(r2)
        results['MSE'].append(mse)
        results['MAE'].append(mae)
        results['Training Time (s)'].append(training_time)
        results['Prediction Time (s)'].append(prediction_time)
        results['Memory Usage (MB)'].append(memory_usage)
        
        # Store model and predictions for visualization
        model_objects[name] = model
        predictions[name] = y_pred
        
        # Update progress
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text('Training completed!')
    
    return results, model_objects, predictions

def create_actual_vs_predicted_plot(y_test, predictions):
    """Create actual vs predicted plots for all models"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(predictions.keys()),
        x_title="Actual Price",
        y_title="Predicted Price"
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        row, col = positions[i]
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_test, 
                y=y_pred, 
                mode='markers',
                name=name,
                opacity=0.6,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Actual vs Predicted Prices")
    return fig

def create_metrics_comparison_plot(results_df):
    """Create comparison plots for metrics"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['R² Score', 'MSE', 'MAE'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R2 Score
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['R2 Score'], name='R² Score', marker_color='blue'),
        row=1, col=1
    )
    
    # MSE
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['MSE'], name='MSE', marker_color='red'),
        row=1, col=2
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['MAE'], name='MAE', marker_color='green'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, title_text="Model Performance Metrics Comparison", showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_performance_comparison_plot(results_df):
    """Create comparison plots for performance metrics"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Training Time', 'Memory Usage'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Training Time
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Training Time (s)'], name='Training Time', marker_color='orange'),
        row=1, col=1
    )
    
    # Memory Usage
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Memory Usage (MB)'], name='Memory Usage', marker_color='purple'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Model Performance Comparison", showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    return fig

def get_complexity_analysis():
    """Get theoretical complexity analysis"""
    complexity = {
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'SVR'],
        'Time Complexity (Training)': ['O(n×p² + p³)', 'O(trees×n×log(n)×p)', 'O(trees×n×log(n)×p)', 'O(n²×p + n³)'],
        'Time Complexity (Prediction)': ['O(p)', 'O(trees×log(n)×p)', 'O(trees×log(n)×p)', 'O(n_sv×p)'],
        'Space Complexity': ['O(p)', 'O(trees×n×p)', 'O(trees×n×p)', 'O(n_sv×p)'],
        'Algorithm Type': ['Linear', 'Ensemble (Bagging)', 'Ensemble (Boosting)', 'Kernel Method'],
        'Interpretability': ['High', 'Medium', 'Low', 'Low'],
        'Overfitting Risk': ['Low', 'Medium', 'High', 'Medium']
    }
    return pd.DataFrame(complexity)

# Home Section
if section == "🏠 Home":
    st.header("Welcome to ML Model Comparison Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Purpose")
        st.write("""
        This application compares four popular machine learning algorithms for house price prediction:
        - **Linear Regression**: Simple, interpretable baseline
        - **Random Forest**: Ensemble method with bagging
        - **Gradient Boosting**: Ensemble method with boosting
        - **Support Vector Regression (SVR)**: Kernel-based method
        """)
        
        st.subheader("📊 Evaluation Metrics")
        st.write("""
        - **R² Score**: Coefficient of determination (higher is better)
        - **MSE**: Mean Squared Error (lower is better)
        - **MAE**: Mean Absolute Error (lower is better)
        - **Training Time**: Time complexity measurement
        - **Memory Usage**: Space complexity measurement
        """)
    
    with col2:
        st.subheader("🗂️ Dataset Information")
        st.write("""
        **California Housing Dataset**
        - 20,640 samples
        - 8 features (location, housing characteristics)
        - Target: Median house price (in hundreds of thousands)
        - Built into scikit-learn
        """)
        
        st.subheader("🚀 Getting Started")
        st.write("""
        1. Navigate to **Dataset Overview** to explore the data
        2. Go to **Model Training** to train models with custom parameters
        3. View **Results & Comparison** for detailed analysis
        4. Check **Complexity Analysis** for theoretical insights
        """)

# Dataset Overview Section
elif section == "📊 Dataset Overview":
    st.header("California Housing Dataset Overview")
    
    X, y, description = load_data()
    
    if X is not None and y is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"**Samples**: {X.shape[0]:,}")
            st.write(f"**Features**: {X.shape[1]}")
            st.write(f"**Target variable**: House prices")
            
            st.subheader("Feature Information")
            st.dataframe(X.describe())
            
        with col2:
            st.subheader("Target Distribution")
            fig = px.histogram(y, nbins=50, title="Distribution of House Prices")
            fig.update_xaxes(title="Price (hundreds of thousands)")
            fig.update_yaxes(title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Feature Correlations")
        correlation_data = pd.concat([X, pd.Series(y, name='Price')], axis=1)
        correlation_matrix = correlation_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sample Data")
        st.dataframe(X.head(10))
        
        with st.expander("Dataset Description"):
            st.text(description)
    else:
        st.error("Failed to load dataset. Please check your internet connection.")

# Model Training Section
elif section == "🔧 Model Training":
    st.header("Model Training Configuration")
    
    X, y, _ = load_data()
    
    if X is not None and y is not None:
        st.subheader("Training Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider("Number of Estimators (RF & GB)", 10, 200, 100, 10)
            
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            
        with col3:
            random_state = st.number_input("Random State", 1, 100, 42)
        
        st.subheader("Model Information")
        
        model_info = {
            "Linear Regression": "Fast, interpretable baseline model with linear assumptions",
            "Random Forest": f"Ensemble of {n_estimators} decision trees using bagging",
            "Gradient Boosting": f"Sequential ensemble of {n_estimators} weak learners",
            "SVR": "Support Vector Regression with RBF kernel for non-linear patterns"
        }
        
        for model, info in model_info.items():
            with st.expander(f"{model}"):
                st.write(info)
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Preparing data..."):
                X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y, test_size, random_state)
                models = get_models(n_estimators, random_state)
            
            st.success("Data prepared successfully!")
            
            with st.spinner("Training models..."):
                results, model_objects, predictions = train_and_evaluate_models(
                    models, X_train_scaled, X_test_scaled, y_train, y_test
                )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.predictions = predictions
            st.session_state.y_test = y_test
            st.session_state.models_trained = True
            st.session_state.complexity_df = get_complexity_analysis()
            
            st.success("🎉 All models trained successfully!")
            st.balloons()
            
            # Show quick results
            results_df = pd.DataFrame(results)
            st.subheader("Quick Results Summary")
            st.dataframe(results_df.round(4))
            
            st.info("Navigate to 'Results & Comparison' section for detailed analysis!")
    else:
        st.error("Failed to load dataset. Please check your internet connection.")

# Results & Comparison Section
elif section == "📈 Results & Comparison":
    st.header("Model Results & Comparison")
    
    if st.session_state.models_trained and st.session_state.results is not None:
        results_df = pd.DataFrame(st.session_state.results)
        
        # Performance metrics table
        st.subheader("📊 Performance Metrics")
        st.dataframe(results_df.round(4))
        
        # Download results
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
        
        # Metrics comparison plots
        st.subheader("📈 Metrics Comparison")
        metrics_fig = create_metrics_comparison_plot(results_df)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Performance comparison plots
        st.subheader("⚡ Performance Comparison")
        performance_fig = create_performance_comparison_plot(results_df)
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Actual vs Predicted plots
        st.subheader("🎯 Actual vs Predicted")
        prediction_fig = create_actual_vs_predicted_plot(st.session_state.y_test, st.session_state.predictions)
        st.plotly_chart(prediction_fig, use_container_width=True)
        
        # Best model analysis
        st.subheader("🏆 Best Model Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_r2_idx = results_df['R2 Score'].idxmax()
            best_r2_model = results_df.loc[best_r2_idx, 'Model']
            st.metric("Best R² Score", f"{results_df.loc[best_r2_idx, 'R2 Score']:.4f}", f"Model: {best_r2_model}")
        
        with col2:
            best_mse_idx = results_df['MSE'].idxmin()
            best_mse_model = results_df.loc[best_mse_idx, 'Model']
            st.metric("Lowest MSE", f"{results_df.loc[best_mse_idx, 'MSE']:.4f}", f"Model: {best_mse_model}")
        
        with col3:
            fastest_idx = results_df['Training Time (s)'].idxmin()
            fastest_model = results_df.loc[fastest_idx, 'Model']
            st.metric("Fastest Training", f"{results_df.loc[fastest_idx, 'Training Time (s)']:.4f}s", f"Model: {fastest_model}")
        
    else:
        st.info("Please train the models first in the 'Model Training' section.")
        if st.button("Go to Model Training"):
            st.rerun()

# Complexity Analysis Section
elif section == "🧮 Complexity Analysis":
    st.header("Algorithm Complexity Analysis")
    
    if st.session_state.complexity_df is not None:
        st.subheader("📚 Theoretical Complexity")
        st.dataframe(st.session_state.complexity_df)
        
        # Download complexity analysis
        csv_buffer = io.StringIO()
        st.session_state.complexity_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Complexity Analysis CSV",
            data=csv_buffer.getvalue(),
            file_name="complexity_analysis.csv",
            mime="text/csv"
        )
        
        # Complexity explanations
        st.subheader("🔍 Complexity Breakdown")
        
        complexity_explanations = {
            "Linear Regression": {
                "Training": "O(n×p² + p³) - Matrix operations dominate",
                "Prediction": "O(p) - Simple dot product",
                "Space": "O(p) - Stores only coefficients",
                "Pros": "Fast, interpretable, low memory",
                "Cons": "Assumes linear relationships"
            },
            "Random Forest": {
                "Training": "O(trees×n×log(n)×p) - Building multiple trees",
                "Prediction": "O(trees×log(n)×p) - Traversing trees",
                "Space": "O(trees×n×p) - Stores all trees",
                "Pros": "Handles non-linearity, reduces overfitting",
                "Cons": "Higher memory usage, less interpretable"
            },
            "Gradient Boosting": {
                "Training": "O(trees×n×log(n)×p) - Sequential tree building",
                "Prediction": "O(trees×log(n)×p) - Sequential prediction",
                "Space": "O(trees×n×p) - Stores all weak learners",
                "Pros": "High accuracy, handles complex patterns",
                "Cons": "Prone to overfitting, computationally expensive"
            },
            "SVR": {
                "Training": "O(n²×p + n³) - Quadratic programming",
                "Prediction": "O(n_sv×p) - Depends on support vectors",
                "Space": "O(n_sv×p) - Stores support vectors",
                "Pros": "Effective in high dimensions, memory efficient",
                "Cons": "Slow training on large datasets"
            }
        }
        
        for model, explanation in complexity_explanations.items():
            with st.expander(f"{model} - Detailed Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Training Complexity**: {explanation['Training']}")
                    st.write(f"**Prediction Complexity**: {explanation['Prediction']}")
                    st.write(f"**Space Complexity**: {explanation['Space']}")
                with col2:
                    st.write(f"**Advantages**: {explanation['Pros']}")
                    st.write(f"**Disadvantages**: {explanation['Cons']}")
        
        if st.session_state.models_trained:
            st.subheader("📊 Empirical vs Theoretical")
            
            results_df = pd.DataFrame(st.session_state.results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Time Ranking (Empirical)**")
                training_time_ranking = results_df.nsmallest(4, 'Training Time (s)')[['Model', 'Training Time (s)']]
                for i, (_, row) in enumerate(training_time_ranking.iterrows(), 1):
                    st.write(f"{i}. {row['Model']}: {row['Training Time (s)']:.4f}s")
            
            with col2:
                st.write("**Memory Usage Ranking (Empirical)**")
                memory_ranking = results_df.nsmallest(4, 'Memory Usage (MB)')[['Model', 'Memory Usage (MB)']]
                for i, (_, row) in enumerate(memory_ranking.iterrows(), 1):
                    st.write(f"{i}. {row['Model']}: {row['Memory Usage (MB)']:.2f}MB")
        
        st.subheader("📝 Notation Guide")
        st.write("""
        - **n**: Number of training samples
        - **p**: Number of features
        - **trees**: Number of trees in ensemble
        - **n_sv**: Number of support vectors
        """)
        
    else:
        complexity_df = get_complexity_analysis()
        st.session_state.complexity_df = complexity_df
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏠 ML Model Comparison Tool | Built with Streamlit</p>
    <p>Compare Linear Regression, Random Forest, Gradient Boosting, and SVR algorithms</p>
</div>
""", unsafe_allow_html=True)
