import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #673ab7;
        font-weight: 700;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🔮 Predictive Analytics & Forecasting Dashboard")
st.markdown("**Build predictive models to forecast future trends using Machine Learning**")
st.markdown("---")

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")
            return None
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar
st.sidebar.header("📁 Data Import")
uploaded_file = st.sidebar.file_uploader(
    "Upload time-series data (CSV/Excel)",
    type=['csv', 'xlsx', 'xls']
)

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.sidebar.success(f"✅ Loaded {len(df)} records")
else:
    st.sidebar.warning("⚠️ Please upload a file to get started")

# Main analysis
if df is not None:
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    # Sidebar - Model Settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Model Settings")
    
    # Select target variable
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_variable = st.sidebar.selectbox(
        "Select target variable to predict:",
        numeric_cols,
        index=numeric_cols.index('Sales') if 'Sales' in numeric_cols else 0
    )
    
    # Select features for prediction
    available_features = [col for col in numeric_cols if col != target_variable]
    
    default_features = ['DayOfWeek', 'Month', 'IsWeekend', 'MarketingSpend']
    default_features = [f for f in default_features if f in available_features]
    
    selected_features = st.sidebar.multiselect(
        "Select features for prediction:",
        available_features,
        default=default_features if default_features else available_features[:min(4, len(available_features))]
    )
    
    if len(selected_features) < 1:
        st.warning("⚠️ Please select at least 1 feature for prediction")
        st.stop()
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select model:",
        ["Linear Regression", "Random Forest", "Both (Compare)"]
    )
    
    # Train/test split
    test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20, 5)
    
    # Forecast period
    forecast_days = st.sidebar.slider("Forecast days ahead", 7, 90, 30, 7)
    
    # Run prediction button
    if st.sidebar.button("🚀 Run Prediction", type="primary"):
        st.session_state.run_prediction = True
    
    if 'run_prediction' not in st.session_state:
        st.session_state.run_prediction = True
    
    if st.session_state.run_prediction:
        
        # Data Overview
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            date_range = (df['Date'].max() - df['Date'].min()).days
            st.metric("Date Range (days)", f"{date_range:,}")
        
        with col3:
            avg_value = df[target_variable].mean()
            st.metric(f"Avg {target_variable}", f"{avg_value:,.2f}")
        
        with col4:
            total_value = df[target_variable].sum()
            st.metric(f"Total {target_variable}", f"{total_value:,.2f}")
        
        st.markdown("---")
        
        # Historical trend
        st.subheader(f"📈 Historical Trend - {target_variable}")
        
        fig_historical = go.Figure()
        fig_historical.add_trace(go.Scatter(
            x=df['Date'],
            y=df[target_variable],
            mode='lines',
            name=target_variable,
            line=dict(color='#667eea', width=2)
        ))
        
        fig_historical.update_layout(
            xaxis_title="Date",
            yaxis_title=target_variable,
            hovermode='x unified',
            plot_bgcolor='white',
            height=400
        )
        
        st.plotly_chart(fig_historical, use_container_width=True)
        
        st.markdown("---")
        
        # Data preprocessing
        st.header("🔧 Data Preprocessing")
        
        with st.expander("View Preprocessing Steps", expanded=False):
            st.markdown("""
            **Steps Applied:**
            1. ✅ Handle missing values
            2. ✅ Feature selection
            3. ✅ Train-test split
            4. ✅ Feature scaling (if needed)
            """)
            
            # Missing values
            missing = df[selected_features + [target_variable]].isnull().sum()
            if missing.sum() > 0:
                st.warning(f"Missing values found: {missing[missing > 0].to_dict()}")
                df = df.dropna(subset=selected_features + [target_variable])
                st.success("Missing values removed")
            else:
                st.success("No missing values found")
        
        # Prepare data
        X = df[selected_features].copy()
        y = df[target_variable].copy()
        
        # Train-test split
        test_size_ratio = test_size / 100
        split_index = int(len(X) * (1 - test_size_ratio))
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        train_dates = df['Date'][:split_index]
        test_dates = df['Date'][split_index:]
        
        st.markdown("---")
        
        # Model training and evaluation
        st.header("🤖 Model Training & Evaluation")
        
        models = {}
        predictions = {}
        metrics = {}
        
        # Linear Regression
        if model_type in ["Linear Regression", "Both (Compare)"]:
            with st.spinner("Training Linear Regression..."):
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_pred = lr_model.predict(X_test)
                
                models['Linear Regression'] = lr_model
                predictions['Linear Regression'] = lr_pred
                
                # Calculate metrics
                metrics['Linear Regression'] = {
                    'MAE': mean_absolute_error(y_test, lr_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                    'R²': r2_score(y_test, lr_pred),
                    'MAPE': np.mean(np.abs((y_test - lr_pred) / y_test)) * 100
                }
        
        # Random Forest
        if model_type in ["Random Forest", "Both (Compare)"]:
            with st.spinner("Training Random Forest..."):
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                
                models['Random Forest'] = rf_model
                predictions['Random Forest'] = rf_pred
                
                # Calculate metrics
                metrics['Random Forest'] = {
                    'MAE': mean_absolute_error(y_test, rf_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                    'R²': r2_score(y_test, rf_pred),
                    'MAPE': np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
                }
        
        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        
        metrics_df = pd.DataFrame(metrics).T
        metrics_df = metrics_df.round(3)
        
        # Color code the best metrics
        def highlight_best(s):
            if s.name in ['MAE', 'RMSE', 'MAPE']:
                is_min = s == s.min()
                return ['background-color: lightgreen' if v else '' for v in is_min]
            else:  # R²
                is_max = s == s.max()
                return ['background-color: lightgreen' if v else '' for v in is_max]
        
        styled_metrics = metrics_df.style.apply(highlight_best, axis=0)
        st.dataframe(styled_metrics, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Metrics Explained:**
            - **MAE** (Mean Absolute Error): Average prediction error (lower is better)
            - **RMSE** (Root Mean Squared Error): Penalizes large errors (lower is better)
            - **R²** (R-squared): How well model explains variance (higher is better, max 1.0)
            - **MAPE** (Mean Absolute Percentage Error): Average % error (lower is better)
            """)
        
        with col2:
            # Best model
            if len(models) > 1:
                best_model_name = min(metrics.keys(), key=lambda x: metrics[x]['RMSE'])
                st.success(f"🏆 **Best Model:** {best_model_name}")
                st.info(f"**R² Score:** {metrics[best_model_name]['R²']:.3f}")
                st.info(f"**RMSE:** {metrics[best_model_name]['RMSE']:.2f}")
            else:
                model_name = list(models.keys())[0]
                st.success(f"✅ **Model:** {model_name}")
                st.info(f"**R² Score:** {metrics[model_name]['R²']:.3f}")
                st.info(f"**RMSE:** {metrics[model_name]['RMSE']:.2f}")
        
        st.markdown("---")
        
        # Predictions visualization
        st.subheader("📈 Actual vs Predicted Values")
        
        for model_name, pred in predictions.items():
            fig_pred = go.Figure()
            
            # Training data
            fig_pred.add_trace(go.Scatter(
                x=train_dates,
                y=y_train,
                mode='lines',
                name='Training Data',
                line=dict(color='lightgray', width=1)
            ))
            
            # Actual test data
            fig_pred.add_trace(go.Scatter(
                x=test_dates,
                y=y_test,
                mode='lines',
                name='Actual',
                line=dict(color='#667eea', width=2)
            ))
            
            # Predictions
            fig_pred.add_trace(go.Scatter(
                x=test_dates,
                y=pred,
                mode='lines',
                name='Predicted',
                line=dict(color='#f5576c', width=2, dash='dash')
            ))
            
            fig_pred.update_layout(
                title=f'{model_name} - Predictions',
                xaxis_title="Date",
                yaxis_title=target_variable,
                hovermode='x unified',
                plot_bgcolor='white',
                height=400
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
        
        st.markdown("---")
        
        # Residual analysis
        st.subheader("📊 Residual Analysis")
        
        col1, col2 = st.columns(2)
        
        for idx, (model_name, pred) in enumerate(predictions.items()):
            residuals = y_test.values - pred
            
            col = col1 if idx == 0 else col2
            
            with col:
                st.markdown(f"**{model_name}**")
                
                # Residual plot
                fig_residual = go.Figure()
                fig_residual.add_trace(go.Scatter(
                    x=pred,
                    y=residuals,
                    mode='markers',
                    marker=dict(color='#667eea', size=6, opacity=0.6),
                    name='Residuals'
                ))
                
                # Zero line
                fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig_residual.update_layout(
                    title='Residual Plot',
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals',
                    plot_bgcolor='white',
                    height=300
                )
                
                st.plotly_chart(fig_residual, use_container_width=True)
        
        st.markdown("---")
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in models:
            st.subheader("🎯 Feature Importance")
            
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance (Random Forest)',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(plot_bgcolor='white', height=400)
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("---")
        
        # Future forecasting
        st.header("🔮 Future Forecast")
        
        # Select best model for forecasting
        if len(models) > 1:
            forecast_model_name = st.selectbox(
                "Select model for forecasting:",
                list(models.keys()),
                index=0
            )
        else:
            forecast_model_name = list(models.keys())[0]
        
        forecast_model = models[forecast_model_name]
        
        # Generate future dates
        last_date = df['Date'].max()
        future_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
        
        # Prepare future features (simplified - using last known values)
        last_features = X.iloc[-1:].copy()
        future_features = pd.concat([last_features] * forecast_days, ignore_index=True)
        
        # Update time-based features if they exist
        if 'DayOfWeek' in future_features.columns:
            future_features['DayOfWeek'] = [d.weekday() for d in future_dates]
        if 'Month' in future_features.columns:
            future_features['Month'] = [d.month for d in future_dates]
        if 'IsWeekend' in future_features.columns:
            future_features['IsWeekend'] = [1 if d.weekday() >= 5 else 0 for d in future_dates]
        if 'DayOfYear' in future_features.columns:
            future_features['DayOfYear'] = [d.timetuple().tm_yday for d in future_dates]
        if 'Quarter' in future_features.columns:
            future_features['Quarter'] = [(d.month-1)//3 + 1 for d in future_dates]
        
        # Make predictions
        future_predictions = forecast_model.predict(future_features)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_' + target_variable: future_predictions
        })
        
        # Visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_forecast = go.Figure()
            
            # Historical data (last 90 days)
            recent_df = df.tail(90)
            fig_forecast.add_trace(go.Scatter(
                x=recent_df['Date'],
                y=recent_df[target_variable],
                mode='lines',
                name='Historical',
                line=dict(color='#667eea', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Predicted_' + target_variable],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#f5576c', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Confidence interval (simplified)
            std_error = metrics[forecast_model_name]['RMSE']
            upper_bound = future_predictions + (1.96 * std_error)
            lower_bound = future_predictions - (1.96 * std_error)
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=upper_bound,
                mode='lines',
                name='Upper Bound (95%)',
                line=dict(color='rgba(245, 87, 108, 0.2)', width=0),
                showlegend=False
            ))
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=lower_bound,
                mode='lines',
                name='Lower Bound (95%)',
                line=dict(color='rgba(245, 87, 108, 0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(245, 87, 108, 0.2)',
                showlegend=True
            ))
            
            fig_forecast.update_layout(
                title=f'{forecast_days}-Day Forecast ({forecast_model_name})',
                xaxis_title="Date",
                yaxis_title=target_variable,
                hovermode='x unified',
                plot_bgcolor='white',
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Forecast Summary")
            
            avg_forecast = future_predictions.mean()
            total_forecast = future_predictions.sum()
            max_forecast = future_predictions.max()
            min_forecast = future_predictions.min()
            
            st.metric("Average Forecast", f"{avg_forecast:,.2f}")
            st.metric("Total Forecast", f"{total_forecast:,.2f}")
            st.metric("Max Value", f"{max_forecast:,.2f}")
            st.metric("Min Value", f"{min_forecast:,.2f}")
            
            # Trend
            if future_predictions[-1] > future_predictions[0]:
                trend = "📈 Upward"
                trend_color = "green"
            else:
                trend = "📉 Downward"
                trend_color = "red"
            
            st.markdown(f"**Trend:** :{trend_color}[{trend}]")
            
            # Growth rate
            growth = ((future_predictions[-1] - df[target_variable].iloc[-1]) / df[target_variable].iloc[-1]) * 100
            st.metric("Projected Growth", f"{growth:+.2f}%")
        
        # Forecast table
        st.subheader("📋 Detailed Forecast")
        
        forecast_display = forecast_df.copy()
        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
        forecast_display['Day'] = [d.strftime('%A') for d in future_dates]
        
        # Add confidence intervals
        forecast_display['Lower_Bound'] = lower_bound
        forecast_display['Upper_Bound'] = upper_bound
        
        # Reorder columns
        forecast_display = forecast_display[['Date', 'Day', 'Predicted_' + target_variable, 'Lower_Bound', 'Upper_Bound']]
        forecast_display = forecast_display.round(2)
        
        st.dataframe(forecast_display, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Business insights
        st.header("💡 Business Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Key Findings")
            
            # Model performance insight
            best_r2 = max([m['R²'] for m in metrics.values()])
            if best_r2 > 0.8:
                model_quality = "Excellent"
                quality_color = "green"
            elif best_r2 > 0.6:
                model_quality = "Good"
                quality_color = "blue"
            else:
                model_quality = "Needs Improvement"
                quality_color = "orange"
            
            st.markdown(f"- **Model Quality:** :{quality_color}[{model_quality}] (R² = {best_r2:.3f})")
            
            # Forecast trend
            st.markdown(f"- **Forecast Trend:** {trend}")
            st.markdown(f"- **Projected Growth:** {growth:+.2f}%")
            
            # Peak day
            peak_idx = np.argmax(future_predictions)
            peak_date = future_dates[peak_idx]
            st.markdown(f"- **Peak Expected:** {peak_date.strftime('%Y-%m-%d (%A)')}")
            
            # Average prediction
            st.markdown(f"- **Daily Average:** {avg_forecast:,.2f}")
        
        with col2:
            st.subheader("📈 Recommendations")
            
            if growth > 0:
                st.success("""
                **Positive Growth Expected:**
                - Prepare inventory for increased demand
                - Consider scaling marketing efforts
                - Ensure adequate staffing levels
                - Review supply chain capacity
                """)
            else:
                st.warning("""
                **Declining Trend Detected:**
                - Investigate potential causes
                - Plan promotional campaigns
                - Review competitive landscape
                - Consider seasonal adjustments
                """)
            
            # Feature-based recommendations
            if 'Random Forest' in models and 'MarketingSpend' in selected_features:
                importance_idx = selected_features.index('MarketingSpend')
                marketing_importance = models['Random Forest'].feature_importances_[importance_idx]
                
                if marketing_importance > 0.2:
                    st.info("💰 Marketing spend shows high importance - consider optimizing budget allocation")
        
        st.markdown("---")
        
        # Export options
        st.header("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export forecast
            csv_forecast = forecast_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast",
                data=csv_forecast,
                file_name=f'forecast_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
        
        with col2:
            # Export metrics
            csv_metrics = metrics_df.to_csv().encode('utf-8')
            st.download_button(
                label="📊 Download Metrics",
                data=csv_metrics,
                file_name=f'model_metrics_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )
        
        with col3:
            # Export predictions
            predictions_df = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_test.values,
                **{f'Predicted_{name}': pred for name, pred in predictions.items()}
            })
            csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="🎯 Download Predictions",
                data=csv_predictions,
                file_name=f'predictions_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

else:
    # Welcome screen
    st.info("👋 **Welcome to Predictive Analytics Dashboard!**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is Predictive Analytics?
        
        Predictive analytics uses historical data and machine learning to forecast future trends:
        - **Sales forecasting**
        - **Demand prediction**
        - **Revenue estimation**
        - **Trend analysis**
        
        ### ✨ Key Features
        
        - 🤖 Multiple ML models (Linear Regression, Random Forest)
        - 📊 Model performance metrics
        - 🔮 Future forecasting
        - 📈 Trend analysis
        - 💡 Business insights
        - 📉 Residual analysis
        - 🎯 Feature importance
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Required Data Format
        
        Your CSV/Excel file should contain:
        
        | Column | Type | Example |
        |--------|------|---------|
        | Date | Date | 2024-01-01 |
        | Sales | Number | 1500.50 |
        | DayOfWeek | Number | 0-6 |
        | Month | Number | 1-12 |
        | IsWeekend | Number | 0 or 1 |
        | MarketingSpend | Number | 500.00 |
        
        **Note:** Additional features can improve prediction accuracy!
        
        ### 📊 Models Available
        
        - **Linear Regression**: Fast, interpretable
        - **Random Forest**: More accurate, handles non-linearity
        """)
    
    st.markdown("---")
    
    st.subheader("📝 Sample Data Preview")
    sample_preview = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Sales': [1500.50, 1620.30, 1450.75],
        'DayOfWeek': [0, 1, 2],
        'Month': [1, 1, 1],
        'IsWeekend': [0, 0, 0],
        'MarketingSpend': [500.00, 550.00, 480.00],
        'Temperature': [62.5, 63.2, 61.8],
        'EconomicIndex': [100.5, 100.7, 100.6]
    })
    st.dataframe(sample_preview, use_container_width=True, hide_index=True)
    
    st.info("💡 **Tip:** Upload your time-series data to start forecasting!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🔮 Predictive Analytics Dashboard | Powered by Machine Learning & Python</p>
        <p>Data Analyst Internship Project - Task 3</p>
    </div>
    """,
    unsafe_allow_html=True
)