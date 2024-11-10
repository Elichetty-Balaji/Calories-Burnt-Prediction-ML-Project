import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Calories Prediction - Linear Regression",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    # Load the datasets
    calories = pd.read_csv('calories.csv')
    exercise = pd.read_csv('exercise.csv')
    
    # Merge the datasets
    data = pd.concat([exercise, calories['Calories']], axis=1)
    
    # Convert categorical variables
    data['Gender'] = data['Gender'].map({'male': 0, 'female': 1})
    
    return data

@st.cache_resource
def train_model(data):
    # Prepare features and target
    X = data[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
    y = data['Calories']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    return model, scaler, X_test, y_test, y_test_pred

def create_feature_importance_plot(model):
    feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    importance = importance.sort_values('Coefficient', key=abs, ascending=False)
    
    fig = px.bar(importance, x='Coefficient', y='Feature', 
                 title='Feature Importance',
                 orientation='h')
    return fig

def create_actual_vs_predicted_plot(y_test, y_pred):
    fig = px.scatter(x=y_test, y=y_pred, 
                    labels={'x': 'Actual Calories', 'y': 'Predicted Calories'},
                    title='Actual vs Predicted Calories')
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                            y=[y_test.min(), y_test.max()],
                            mode='lines', 
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')))
    return fig

def main():
    st.title("🔥 Calories Burnt Prediction using Linear Regression")
    st.markdown("""
    This application uses Linear Regression to predict the number of calories burned during exercise 
    based on various physical and exercise-related parameters.
    """)
    
    # Load and train model
    with st.spinner("Loading and training model..."):
        data = load_and_preprocess_data()
        model, scaler, X_test, y_test, y_pred = train_model(data)
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select a Page",
        ["Make Prediction", "Model Analysis", "Data Exploration"]
    )
    
    if page == "Make Prediction":
        st.header("Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=15, max_value=80, value=30)
            height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0)
        
        with col2:
            duration = st.number_input("Exercise Duration (minutes)", min_value=1.0, max_value=60.0, value=15.0)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=60.0, max_value=130.0, value=90.0)
            body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=41.0, value=38.0)
        
        if st.button("Predict Calories"):
            # Prepare input data
            input_data = pd.DataFrame([[
                0 if gender == "Male" else 1,
                age,
                height,
                weight,
                duration,
                heart_rate,
                body_temp
            ]], columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.success(f"Predicted Calories Burnt: {prediction:.2f} calories")
            
            # Show calculation details
            with st.expander("See calculation details"):
                st.write("Linear Regression Equation:")
                equation = "Calories = "
                for coef, feature in zip(model.coef_, input_data.columns):
                    equation += f"{coef:.2f}*{feature} + "
                equation += f"{model.intercept_:.2f}"
                st.write(equation)
    
    elif page == "Model Analysis":
        st.header("Model Analysis")
        
        # Model metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
        
        with metrics_col2:
            st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.3f}")
        
        with metrics_col3:
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.3f}")
        
        # Plots
        st.subheader("Feature Importance")
        st.plotly_chart(create_feature_importance_plot(model), use_container_width=True)
        
        st.subheader("Actual vs Predicted Values")
        st.plotly_chart(create_actual_vs_predicted_plot(y_test, y_pred), use_container_width=True)
    
    else:  # Data Exploration
        st.header("Data Exploration")
        
        st.subheader("Sample Data")
        st.dataframe(data.head())
        
        st.subheader("Data Distribution")
        feature = st.selectbox("Select Feature", data.columns[1:])
        fig = px.histogram(data, x=feature, title=f'Distribution of {feature}')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Matrix")
        corr = data.corr()
        fig = px.imshow(corr, 
                       labels=dict(color="Correlation"),
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
