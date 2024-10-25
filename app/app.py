import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import time
from typing import Tuple, Optional
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Virus Sequence Clustering Analysis",
    page_icon=":dna:",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress .st-bo {
        background-color: #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'has_headers' not in st.session_state:
    st.session_state.has_headers = True
if 'features_selected' not in st.session_state:
    st.session_state.features_selected = False
if 'columns_renamed' not in st.session_state:
    st.session_state.columns_renamed = False
if 'categorical_encoded' not in st.session_state:
    st.session_state.categorical_encoded = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'renamed_cols' not in st.session_state:
    st.session_state.renamed_cols = {}

# Callback functions
def on_next():
    st.session_state.current_step += 1

def on_prev():
    st.session_state.current_step -= 1

def skip_to_step(step):
    st.session_state.current_step = step

def main():
    st.title("Virus Sequence Clustering Analysis")
    
    # Progress indicator
    st.progress(st.session_state.current_step / 5)
    
    # Step 1: File Upload
    if st.session_state.current_step == 1:
        st.header("Step 1: Upload Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
        
        if uploaded_file is not None:
            try:
                # Read the data
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.session_state.data_loaded = True
                
                # Show data preview
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                # Show basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Create columns for button alignment
                col1, col2, col3 = st.columns([3, 1, 3])
                with col2:
                    st.button("Next →", type="primary", key="next_step1", on_click=on_next)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.info("Please upload a CSV file to continue.")
    
    # Step 2: Feature Selection and Headers
    elif st.session_state.current_step == 2:
        st.header("Step 2: Feature Selection and Column Names")
        
        # Header selection
        has_headers = st.radio("Does your CSV file have headers?", ['Yes', 'No'], 
                             index=0 if st.session_state.has_headers else 1,
                             key="header_radio")
        st.session_state.has_headers = (has_headers == 'Yes')
        
        # Feature selection
        if not st.session_state.has_headers:
            # If no headers, assign default names
            st.session_state.data.columns = [f'Column_{i+1}' for i in range(st.session_state.data.shape[1])]
        
        selected_features = st.multiselect(
            "Select feature columns for clustering:",
            st.session_state.data.columns,
            default=[col for col in st.session_state.data.columns if col not in ['id', 'label', 'target']],
            key="feature_select"
        )
        
        if selected_features:
            st.session_state.selected_features = selected_features
            st.session_state.features_selected = True
            
            # Create columns for button alignment
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            with col2:
                st.button("← Previous", key="prev_step2", on_click=on_prev)
            with col3:
                if st.session_state.has_headers:
                    st.button("Next →", type="primary", key="next_step2", 
                             on_click=skip_to_step, args=(4,))  # Skip to step 4
                else:
                    st.button("Next →", type="primary", key="next_step2", on_click=on_next)
        else:
            st.warning("Please select at least one feature to continue.")
    
    # Step 3: Column Renaming (only if no headers)
    elif st.session_state.current_step == 3:
        st.header("Step 3: Column Renaming")
        
        with st.expander("Rename Columns", expanded=True):
            st.write("Provide names for selected columns:")
            
            renamed_cols = {}
            cols = st.columns(3)
            
            for i, col in enumerate(st.session_state.selected_features):
                with cols[i % 3]:
                    new_name = st.text_input(
                        f"Name for {col}:",
                        key=f"rename_{col}",
                        value=col
                    ).strip()
                    if new_name:
                        renamed_cols[col] = new_name
            
            if st.button("Apply Names", key="apply_names"):
                st.session_state.renamed_cols = renamed_cols
                st.session_state.columns_renamed = True
                st.session_state.data = st.session_state.data.rename(columns=renamed_cols)
                st.success("Column names updated!")
        
        # Create columns for button alignment
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        with col2:
            st.button("← Previous", key="prev_step3", on_click=on_prev)
        with col3:
            st.button("Next →", type="primary", key="next_step3", on_click=on_next)
    
    # Step 4: Categorical Encoding
    elif st.session_state.current_step == 4:
        st.header("Step 4: Categorical Columns Encoding")
        
        with st.expander("Encode Categorical Columns", expanded=True):
            # Get categorical columns from selected features
            categorical_cols = st.session_state.data[st.session_state.selected_features].select_dtypes(
                include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                st.write("Select encoding method for categorical columns:")
                encoding_methods = {}
                
                for col in categorical_cols:
                    encoding_methods[col] = st.selectbox(
                        f"Encoding method for {col}:",
                        ['Label Encoding', 'One-Hot Encoding'],
                        key=f"encode_{col}"
                    )
                
                if st.button("Apply Encoding", key="apply_encoding"):
                    processed_df = st.session_state.data[st.session_state.selected_features].copy()
                    
                    for col in categorical_cols:
                        if encoding_methods[col] == 'Label Encoding':
                            processed_df[col] = processed_df[col].astype('category').cat.codes
                        else:  # One-Hot Encoding
                            dummies = pd.get_dummies(processed_df[col], prefix=col)
                            processed_df = pd.concat([processed_df, dummies], axis=1)
                            processed_df.drop(col, axis=1, inplace=True)
                    
                    st.session_state.processed_data = processed_df
                    st.session_state.categorical_encoded = True
                    st.success("Categorical encoding completed!")
            else:
                st.info("No categorical columns found in selected features.")
                st.session_state.categorical_encoded = True
        
        # Create columns for button alignment
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        with col2:
            st.button("← Previous", key="prev_step4", on_click=on_prev)
        with col3:
            if st.session_state.categorical_encoded:
                st.button("Next →", type="primary", key="next_step4", on_click=on_next)
    
    # Step 5: Clustering Analysis
    elif st.session_state.current_step == 5:
        st.header("Step 5: Clustering Analysis")
        
        with st.form("clustering_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                n_clusters = st.slider("Number of clusters (k)", 2, 10, 5)
                perplexity = st.slider("t-SNE perplexity", 5, 100, 30)
            
            with col2:
                tsne_iter = st.slider("t-SNE iterations", 250, 2000, 1000)
                random_state = st.number_input("Random seed", 0, 100, 42)

            cluster_button = st.form_submit_button("Perform Clustering")

        if cluster_button:
            with st.spinner('Performing clustering analysis...'):
                # Progress bar
                progress_bar = st.progress(0)
                
                # Get data for clustering
                if st.session_state.processed_data is not None:
                    X = st.session_state.processed_data[st.session_state.selected_features].values
                else:
                    X = st.session_state.data[st.session_state.selected_features].values
                
                # PCA
                progress_bar.progress(20)
                pca = PCA(n_components=2, random_state=random_state)
                X_pca = pca.fit_transform(X)
                
                # K-means
                progress_bar.progress(40)
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
                clusters = kmeans.fit_predict(X)
                
                # t-SNE
                progress_bar.progress(60)
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    n_iter=tsne_iter,
                    random_state=random_state
                )
                X_tsne = tsne.fit_transform(X)
                
                progress_bar.progress(100)

            # Visualization Section
            st.header("Visualization")
            
            tab1, tab2, tab3 = st.tabs(["t-SNE", "PCA", "Cluster Statistics"])
            
            with tab1:
                # t-SNE plot
                fig_tsne = px.scatter(
                    x=X_tsne[:, 0], y=X_tsne[:, 1],
                    color=[f"Cluster {c}" for c in clusters],
                    title="t-SNE Visualization of Clusters",
                    labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                    height=800
                )
                st.plotly_chart(fig_tsne, use_container_width=True)

            with tab2:
                # PCA plot
                fig_pca = px.scatter(
                    x=X_pca[:, 0], y=X_pca[:, 1],
                    color=[f"Cluster {c}" for c in clusters],
                    title="PCA Visualization of Clusters",
                    labels={"x": "PC1", "y": "PC2"},
                    height=800
                )
                st.plotly_chart(fig_pca, use_container_width=True)

            with tab3:
                # Cluster statistics
                cluster_stats = pd.DataFrame({
                    'Cluster': range(n_clusters),
                    'Size': [sum(clusters == i) for i in range(n_clusters)],
                    'Percentage': [sum(clusters == i)/len(clusters)*100 for i in range(n_clusters)]
                })
                
                st.subheader("Cluster Distribution")
                fig_dist = px.bar(
                    cluster_stats,
                    x='Cluster',
                    y='Size',
                    title="Cluster Sizes",
                    height=600
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.dataframe(cluster_stats, use_container_width=True)

            # Download section
            st.header("Download Results")
            
            # Create result DataFrame with selected features and cluster labels
            result_df = pd.DataFrame(X, columns=st.session_state.selected_features)
            result_df['Cluster'] = clusters
            
            # Convert to CSV
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download clustered data as CSV",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv"
            )
        
        # Create columns for button alignment
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        with col2:
            st.button("← Previous", key="prev_step5", on_click=on_prev)

if __name__ == "__main__":
    main()