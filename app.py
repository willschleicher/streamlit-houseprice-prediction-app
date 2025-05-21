import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# Load the models
kmeans_model = joblib.load('kmeans_model.pkl')
lr_model = joblib.load('linear_regression_model.pkl')
cluster_scaler = joblib.load('cluster_scaler.pkl')

# Load a sample of the data for visualization
train_data = pd.read_csv('train.csv')

# Add the engineered features to the data
train_data['TotalSF'] = train_data['1stFlrSF'] + train_data['2ndFlrSF'] + train_data['TotalBsmtSF']
train_data['Age'] = 2010 - train_data['YearBuilt']
train_data['Remodeled'] = (train_data['YearRemodAdd'] != train_data['YearBuilt']).astype(int)
train_data['TotalBathrooms'] = train_data['FullBath'] + (0.5 * train_data['HalfBath']) + \
                             train_data['BsmtFullBath'] + (0.5 * train_data['BsmtHalfBath'])

# Set page configuration
st.set_page_config(
    page_title="Acme Realty Analytics - Housing Price Prediction",
    layout="wide"
)

# Add title and description
st.title("Acme Realty Analytics - Housing Price Prediction Tool")
st.markdown("""
This application helps real estate agents quickly estimate property values
based on key features rather than spending hours on manual market analysis.
""")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Price Prediction", "Market Analysis", "Data Explorer"]
)

# Home page
if page == "Home":
    st.header("Welcome to Acme Realty Analytics")
    st.write("""
    This tool helps real estate agents estimate house prices based on key features.
    
    Use the sidebar to navigate:
    - **Price Prediction**: Get instant price estimates for properties
    - **Market Analysis**: View market segments and trends
    - **Data Explorer**: Explore housing data and patterns
    """)
    
    # Sample visualization for home page
    st.subheader("Housing Price Distribution")
    fig = px.histogram(train_data, x="SalePrice", nbins=50,
                      title="Distribution of House Prices")
    st.plotly_chart(fig)

# Price Prediction page
elif page == "Price Prediction":
    st.header("Housing Price Prediction")
    
    st.write("""
    Enter property details below to get an instant price estimate.
    """)
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        # Input form - left column
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
        gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
        total_sf = st.number_input("Total Square Footage", 500, 10000, 2000)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    
    with col2:
        # Input form - right column
        total_bathrooms = st.slider("Total Bathrooms", 0.0, 6.0, 2.5, 0.5)
        age = st.slider("House Age (years)", 0, 100, 20)
        remodeled = st.selectbox("Recently Remodeled", ["No", "Yes"])
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
    
    # Convert remodeled to binary
    remodeled_binary = 1 if remodeled == "Yes" else 0
    
    # Create input data for prediction
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'TotalSF': [total_sf],
        'GarageCars': [garage_cars],
        'TotalBathrooms': [total_bathrooms],
        'Age': [age],
        'Remodeled': [remodeled_binary],
        'BedroomAbvGr': [bedrooms]
    })
    
    # Make prediction when button is pressed
    if st.button("Predict Price"):
        # Predict price
        predicted_price = lr_model.predict(input_data)[0]
        
        # Determine market segment
        cluster_features = ['OverallQual', 'GrLivArea', 'TotalSF', 'BedroomAbvGr']
        X_cluster = input_data[cluster_features]
        X_cluster_scaled = cluster_scaler.transform(X_cluster)
        cluster = kmeans_model.predict(X_cluster_scaled)[0]
        
        # Find average prices for each cluster
        cluster_features = ['OverallQual', 'GrLivArea', 'TotalSF', 'BedroomAbvGr']
        X_cluster_all = train_data[cluster_features]
        train_data['Cluster'] = kmeans_model.predict(cluster_scaler.transform(X_cluster_all))
        
        cluster_avg_prices = train_data.groupby('Cluster')['SalePrice'].mean()
        
        # Map clusters to market segments
        lowest_price_cluster = cluster_avg_prices.idxmin()
        highest_price_cluster = cluster_avg_prices.idxmax()
        middle_clusters = [i for i in range(3) if i != lowest_price_cluster and i != highest_price_cluster]
        middle_price_cluster = middle_clusters[0] if middle_clusters else None
        
        cluster_names = {
            lowest_price_cluster: 'Economy',
            highest_price_cluster: 'Luxury'
        }
        if middle_price_cluster is not None:
            cluster_names[middle_price_cluster] = 'Mid-Range'
        
        market_segment = cluster_names.get(cluster, "Unknown")
        
        # Display results
        st.subheader("Price Prediction Results")
        
        col1, col2 = st.columns(2)
        col1.metric("Estimated Price", f"${predicted_price:,.2f}")
        col2.metric("Market Segment", market_segment)
        
        st.success(f"This property is estimated to be worth ${predicted_price:,.2f} and falls in the {market_segment} market segment.")
        
        # Compare with similar properties
        st.subheader("Comparison with Similar Properties")
        
        similar_props = train_data[
            (train_data['BedroomAbvGr'] == bedrooms) & 
            (train_data['OverallQual'] == overall_qual)
        ]
        
        if len(similar_props) > 0:
            avg_similar_price = similar_props['SalePrice'].mean()
            st.write(f"Similar {bedrooms}-bedroom properties with quality rating {overall_qual} have an average price of ${avg_similar_price:,.2f}.")
            
            diff = predicted_price - avg_similar_price
            percent_diff = (diff / avg_similar_price) * 100
            
            if diff > 0:
                st.write(f"This property is estimated to be ${diff:,.2f} ({percent_diff:.1f}%) more expensive than similar properties.")
            else:
                st.write(f"This property is estimated to be ${-diff:,.2f} ({-percent_diff:.1f}%) less expensive than similar properties.")

# Market Analysis page
elif page == "Market Analysis":
    st.header("Housing Market Segmentation")
    
    st.write("""
    We've segmented the housing market into three categories based on key features:
    - **Economy**: Lower-priced properties with basic features
    - **Mid-Range**: Average properties with standard features
    - **Luxury**: High-end properties with premium features
    """)
    
    # Apply clustering to the data
    cluster_features = ['OverallQual', 'GrLivArea', 'TotalSF', 'BedroomAbvGr']
    X_cluster = train_data[cluster_features]
    X_cluster_scaled = cluster_scaler.transform(X_cluster)
    train_data['Cluster'] = kmeans_model.predict(X_cluster_scaled)
    
    # Map clusters to market segments
    cluster_avg_prices = train_data.groupby('Cluster')['SalePrice'].mean()
    
    lowest_price_cluster = cluster_avg_prices.idxmin()
    highest_price_cluster = cluster_avg_prices.idxmax()
    middle_clusters = [i for i in range(3) if i != lowest_price_cluster and i != highest_price_cluster]
    middle_price_cluster = middle_clusters[0] if middle_clusters else None
    
    cluster_names = {
        lowest_price_cluster: 'Economy',
        highest_price_cluster: 'Luxury'
    }
    if middle_price_cluster is not None:
        cluster_names[middle_price_cluster] = 'Mid-Range'
    
    train_data['Market_Segment'] = train_data['Cluster'].map(cluster_names)
    
    # VISUALIZATION 1: Scatter plot of clusters
    st.subheader("Market Segments by Living Area and Price")
    fig1 = px.scatter(train_data, x="GrLivArea", y="SalePrice", 
                     color="Market_Segment", size="OverallQual",
                     hover_data=["BedroomAbvGr", "TotalBathrooms", "Age"],
                     title="Housing Market Segments")
    st.plotly_chart(fig1)
    
    # VISUALIZATION 2: Price distribution by segment
    st.subheader("Price Distribution by Market Segment")
    fig2 = px.box(train_data, x="Market_Segment", y="SalePrice", 
                 color="Market_Segment",
                 title="Price Ranges by Market Segment")
    st.plotly_chart(fig2)
    
    # VISUALIZATION 3: Market segment characteristics
    st.subheader("Market Segment Characteristics")
    
    segment_stats = train_data.groupby('Market_Segment').agg({
        'SalePrice': 'mean',
        'OverallQual': 'mean',
        'GrLivArea': 'mean',
        'TotalSF': 'mean',
        'Age': 'mean',
        'TotalBathrooms': 'mean',
        'BedroomAbvGr': 'mean'
    }).reset_index()
    
    fig3 = px.bar(segment_stats, x="Market_Segment", y=["OverallQual", "TotalBathrooms", "BedroomAbvGr"],
                 title="Average Features by Market Segment",
                 barmode="group")
    st.plotly_chart(fig3)

# Data Explorer page
elif page == "Data Explorer":
    st.header("Housing Data Explorer")
    
    st.write("""
    Explore relationships between different housing features and prices.
    """)
    
    # Feature selection
    x_feature = st.selectbox(
        "Select X-axis Feature",
        options=["OverallQual", "GrLivArea", "TotalSF", "Age", "TotalBathrooms", "BedroomAbvGr"]
    )
    
    y_feature = st.selectbox(
        "Select Y-axis Feature",
        options=["SalePrice", "OverallQual", "GrLivArea", "TotalSF"],
        index=0
    )
    
    color_by = st.selectbox(
        "Color by",
        options=["Market_Segment", "OverallQual", "Age"]
    )
    
    # Apply clustering if we need Market_Segment
    if "Market_Segment" not in train_data.columns or color_by == "Market_Segment":
        # Cluster the data
        cluster_features = ['OverallQual', 'GrLivArea', 'TotalSF', 'BedroomAbvGr']
        X_cluster = train_data[cluster_features]
        X_cluster_scaled = cluster_scaler.transform(X_cluster)
        train_data['Cluster'] = kmeans_model.predict(X_cluster_scaled)
        
        # Map clusters to market segments
        cluster_avg_prices = train_data.groupby('Cluster')['SalePrice'].mean()
        
        lowest_price_cluster = cluster_avg_prices.idxmin()
        highest_price_cluster = cluster_avg_prices.idxmax()
        middle_clusters = [i for i in range(3) if i != lowest_price_cluster and i != highest_price_cluster]
        middle_price_cluster = middle_clusters[0] if middle_clusters else None
        
        cluster_names = {
            lowest_price_cluster: 'Economy',
            highest_price_cluster: 'Luxury'
        }
        if middle_price_cluster is not None:
            cluster_names[middle_price_cluster] = 'Mid-Range'
        
        train_data['Market_Segment'] = train_data['Cluster'].map(cluster_names)
    
    # Create scatter plot
    fig = px.scatter(train_data, x=x_feature, y=y_feature, 
                    color=color_by, hover_data=["SalePrice", "BedroomAbvGr", "Age"],
                    title=f"{y_feature} vs {x_feature} by {color_by}")
    
    st.plotly_chart(fig)
    
    # Show correlation heat map
    st.subheader("Feature Correlation Heatmap")
    
    selected_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalSF', 
                         'GarageCars', 'TotalBathrooms', 'Age', 'BedroomAbvGr']
    
    corr = train_data[selected_features].corr()
    
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                   title="Feature Correlation Heatmap")
    st.plotly_chart(fig)
