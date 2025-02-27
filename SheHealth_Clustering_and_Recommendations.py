import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)  # Load CSV file
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Load dataset
file_path = "C:\\Users\\juika\\OneDrive\\Desktop\\SheHealth\\SyntheticDataset.csv" 
df = load_dataset(file_path)

if df is None:
    raise SystemExit("Dataset loading failed. Please check the file path and try again.")

# Display dataset information
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nSample data:")
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())

# Data Preprocessing
print("\n--- Data Preprocessing ---")

# 1. Identify numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# 2. Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Fit and transform the data
processed_data = preprocessor.fit_transform(df)

# Get feature names after one-hot encoding
cat_feature_names = []
if categorical_features:
    onehotencoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = onehotencoder.get_feature_names_out(categorical_features).tolist()

# Combine numerical and categorical feature names
feature_names = numerical_features + cat_feature_names

# Convert to DataFrame for better handling
processed_df = pd.DataFrame(processed_data.toarray(), columns=feature_names)  # Use .toarray() for sparse matrices

print("\nProcessed data shape:", processed_df.shape)
print("Processed data sample:")
print(processed_df.head())

# Split the data into training and testing sets
X_train, X_test = train_test_split(processed_df, test_size=0.2, random_state=42)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Determine optimal number of clusters
print("\n--- Determining Optimal Number of Clusters ---")

# Plot dendrogram for hierarchical clustering
plt.figure(figsize=(12, 8))
dendrogram = shc.dendrogram(shc.linkage(X_train.sample(min(100, len(X_train))), method='ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.axhline(y=6, color='r', linestyle='--')
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()

# Silhouette analysis for different cluster numbers
silhouette_scores_hc = []
silhouette_scores_gmm = []
davies_bouldin_scores_hc = []
davies_bouldin_scores_gmm = []
range_n_clusters = range(2, 8)

for n_clusters in range_n_clusters:
    # Hierarchical Clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels_hc = hc.fit_predict(X_train)
    silhouette_avg_hc = silhouette_score(X_train, cluster_labels_hc)
    db_score_hc = davies_bouldin_score(X_train, cluster_labels_hc)
    silhouette_scores_hc.append(silhouette_avg_hc)
    davies_bouldin_scores_hc.append(db_score_hc)
    
    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels_gmm = gmm.fit_predict(X_train)
    silhouette_avg_gmm = silhouette_score(X_train, cluster_labels_gmm)
    db_score_gmm = davies_bouldin_score(X_train, cluster_labels_gmm)
    silhouette_scores_gmm.append(silhouette_avg_gmm)
    davies_bouldin_scores_gmm.append(db_score_gmm)
    
    print(f"For n_clusters = {n_clusters}:")
    print(f"  HC - Silhouette Score: {silhouette_avg_hc:.3f}, Davies-Bouldin: {db_score_hc:.3f}")
    print(f"  GMM - Silhouette Score: {silhouette_avg_gmm:.3f}, Davies-Bouldin: {db_score_gmm:.3f}")

# Plot silhouette scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, silhouette_scores_hc, 'o-', label='Hierarchical Clustering')
plt.plot(range_n_clusters, silhouette_scores_gmm, 's-', label='GMM')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.legend()
plt.grid(True)

# Plot Davies-Bouldin scores
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, davies_bouldin_scores_hc, 'o-', label='Hierarchical Clustering')
plt.plot(range_n_clusters, davies_bouldin_scores_gmm, 's-', label='GMM')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index vs. Number of Clusters')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('clustering_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# Based on the metrics, select the optimal number of clusters
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores_gmm)]
print(f"\nOptimal number of clusters based on silhouette score: {optimal_n_clusters}")

# Apply the selected models with optimal number of clusters
print("\n--- Applying Clustering Models ---")

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=optimal_n_clusters)
hc_labels_train = hc.fit_predict(X_train)
hc_train_silhouette = silhouette_score(X_train, hc_labels_train)
hc_train_db = davies_bouldin_score(X_train, hc_labels_train)

# GMM
gmm = GaussianMixture(n_components=optimal_n_clusters, random_state=42)
gmm.fit(X_train)
gmm_labels_train = gmm.predict(X_train)
gmm_train_silhouette = silhouette_score(X_train, gmm_labels_train)
gmm_train_db = davies_bouldin_score(X_train, gmm_labels_train)

# Apply to test data
hc_labels_test = hc.fit_predict(X_test)
hc_test_silhouette = silhouette_score(X_test, hc_labels_test)
hc_test_db = davies_bouldin_score(X_test, hc_labels_test)

gmm_labels_test = gmm.predict(X_test)
gmm_test_silhouette = silhouette_score(X_test, gmm_labels_test)
gmm_test_db = davies_bouldin_score(X_test, gmm_labels_test)

print("\nHierarchical Clustering Results:")
print(f"  Training - Silhouette: {hc_train_silhouette:.3f}, Davies-Bouldin: {hc_train_db:.3f}")
print(f"  Testing - Silhouette: {hc_test_silhouette:.3f}, Davies-Bouldin: {hc_test_db:.3f}")

print("\nGaussian Mixture Model Results:")
print(f"  Training - Silhouette: {gmm_train_silhouette:.3f}, Davies-Bouldin: {gmm_train_db:.3f}")
print(f"  Testing - Silhouette: {gmm_test_silhouette:.3f}, Davies-Bouldin: {gmm_test_db:.3f}")

# Select the best model based on test metrics
if gmm_test_silhouette > hc_test_silhouette:
    best_model = "GMM"
    best_labels = gmm_labels_test
    best_train_labels = gmm_labels_train
else:
    best_model = "Hierarchical Clustering"
    best_labels = hc_labels_test
    best_train_labels = hc_labels_train

print(f"\nBest model based on silhouette score: {best_model}")

# Visualize clusters
print("\n--- Visualizing Clusters ---")

# PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, optimal_n_clusters))
cmap = ListedColormap(colors)

# Plot training clusters
plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=best_train_labels, cmap=cmap, s=50, alpha=0.8)
plt.title(f'Training Data Clusters using {best_model}')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')

# Plot testing clusters
plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=best_labels, cmap=cmap, s=50, alpha=0.8)
plt.title(f'Testing Data Clusters using {best_model}')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Generating Cluster Statistics ---
print("\n--- Generating Cluster Statistics ---")

# Create df_train and df_test by slicing the original DataFrame (df) using the indices of X_train and X_test
df_train = df.loc[X_train.index]  # Use the indices of X_train to get the corresponding rows in df
df_test = df.loc[X_test.index]    # Use the indices of X_test to get the corresponding rows in df

# Add cluster labels to df_train
df_train['cluster'] = best_train_labels

# Verify the cluster column
print("\nCluster column in df_train:")
print(df_train['cluster'].head())

# Now, calculate cluster statistics for numeric columns
numeric_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Ensure 'cluster' is not included in numeric_columns
if 'cluster' in numeric_columns:
    numeric_columns.remove('cluster')

# Group by cluster and calculate mean
cluster_stats = df_train[numeric_columns + ['cluster']].groupby('cluster').mean()

print("\nCluster Statistics:")
print(cluster_stats)

# Visualize feature distribution across clusters
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features):
    if i < 9:  # Plot 9 features
        plt.subplot(3, 3, i+1)
        for cluster in range(optimal_n_clusters):
            sns.kdeplot(df_train[df_train['cluster'] == cluster][feature], 
                        label=f'Cluster {cluster}')
        plt.title(f'Distribution of {feature}')
        plt.legend()
plt.tight_layout()
plt.savefig('cluster_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize cluster characteristics using radar chart
print("\n--- Creating Radar Chart for Cluster Profiles ---")

# Normalize cluster means for radar chart
scaler = MinMaxScaler()
normalized_stats = pd.DataFrame(
    scaler.fit_transform(cluster_stats[numerical_features]),
    columns=numerical_features,
    index=cluster_stats.index
)

# Create radar chart
categories = numerical_features
N = len(categories)

# Set up the angles of the radar chart
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, polar=True)

for cluster in range(optimal_n_clusters):
    values = normalized_stats.iloc[cluster].values.flatten().tolist()
    values += values[:1]  # Close the loop
    
    # Plot the cluster
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
    ax.fill(angles, values, alpha=0.1)

# Set the labels and styling
plt.xticks(angles[:-1], categories, size=12)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], color='grey', size=10)
plt.ylim(0, 1)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Cluster Profiles', size=20, y=1.1)
plt.savefig('cluster_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate health recommendations based on clusters
print("\n--- Generating Health Recommendations ---")

# Define recommendations based on cluster characteristics
recommendations = {}

for cluster in range(optimal_n_clusters):
    cluster_info = cluster_stats.loc[cluster]
    
    recommendations[cluster] = {
        'dietary_recommendations': [],
        'exercise_recommendations': [],
        'lifestyle_recommendations': [],
        'monitoring_recommendations': []
    }
    
    # Dietary recommendations based on BMI
    if 'bmi' in cluster_info and cluster_info['bmi'] > 30:
        recommendations[cluster]['dietary_recommendations'].append("Focus on calorie deficit diet with high protein")
        recommendations[cluster]['dietary_recommendations'].append("Consult with nutritionist for personalized meal plan")
    elif 'bmi' in cluster_info and 25 <= cluster_info['bmi'] <= 30:
        recommendations[cluster]['dietary_recommendations'].append("Balanced diet with portion control")
        recommendations[cluster]['dietary_recommendations'].append("Increase intake of fruits and vegetables")
    else:
        recommendations[cluster]['dietary_recommendations'].append("Maintain balanced nutritional intake")
        recommendations[cluster]['dietary_recommendations'].append("Ensure adequate protein intake")
    
    # Exercise recommendations based on BMI and exercise frequency
    if 'exerciseFrequency' in cluster_info and cluster_info['exerciseFrequency'] < 3:
        recommendations[cluster]['exercise_recommendations'].append("Gradually increase physical activity to at least 150 minutes/week")
        recommendations[cluster]['exercise_recommendations'].append("Include both cardio and strength training")
    else:
        recommendations[cluster]['exercise_recommendations'].append("Maintain current exercise routine with focus on variety")
        recommendations[cluster]['exercise_recommendations'].append("Consider adding flexibility and balance exercises")
    
    # Lifestyle recommendations based on stress level and sleep
    if 'stressLevel' in cluster_info and cluster_info['stressLevel'] > 7:
        recommendations[cluster]['lifestyle_recommendations'].append("Implement stress management techniques like meditation or yoga")
        recommendations[cluster]['lifestyle_recommendations'].append("Consider seeking mental health support")
    
    if 'sleepHours' in cluster_info and cluster_info['sleepHours'] < 7:
        recommendations[cluster]['lifestyle_recommendations'].append("Improve sleep hygiene and aim for 7-9 hours of sleep")
        recommendations[cluster]['lifestyle_recommendations'].append("Establish a consistent sleep schedule")
    
    # Monitoring recommendations based on blood pressure
    if 'systolic_bp' in cluster_info and cluster_info['systolic_bp'] > 130:
        recommendations[cluster]['monitoring_recommendations'].append("Regular blood pressure monitoring")
        recommendations[cluster]['monitoring_recommendations'].append("Consult healthcare provider about hypertension management")
    
    # General recommendations for all
    recommendations[cluster]['monitoring_recommendations'].append("Annual comprehensive health checkups")

# Display recommendations for each cluster
for cluster, recs in recommendations.items():
    print(f"\nRecommendations for Cluster {cluster}:")
    print("  Dietary Recommendations:")
    for rec in recs['dietary_recommendations']:
        print(f"    - {rec}")
    
    print("  Exercise Recommendations:")
    for rec in recs['exercise_recommendations']:
        print(f"    - {rec}")
    
    print("  Lifestyle Recommendations:")
    for rec in recs['lifestyle_recommendations']:
        print(f"    - {rec}")
    
    print("  Monitoring Recommendations:")
    for rec in recs['monitoring_recommendations']:
        print(f"    - {rec}")

# Create a recommendation visualization
plt.figure(figsize=(12, 8))
recommendation_types = ['Dietary', 'Exercise', 'Lifestyle', 'Monitoring']
recommendation_counts = []

for cluster in range(optimal_n_clusters):
    cluster_counts = [
        len(recommendations[cluster]['dietary_recommendations']),
        len(recommendations[cluster]['exercise_recommendations']),
        len(recommendations[cluster]['lifestyle_recommendations']),
        len(recommendations[cluster]['monitoring_recommendations'])
    ]
    recommendation_counts.append(cluster_counts)

recommendation_counts = np.array(recommendation_counts).T

# Create stacked bar chart
bar_width = 0.5
bars = []
bottom = np.zeros(optimal_n_clusters)

for i, counts in enumerate(recommendation_counts):
    p = plt.bar(range(optimal_n_clusters), counts, bar_width, bottom=bottom, label=recommendation_types[i])
    bottom += counts
    bars.append(p)

plt.xlabel('Cluster')
plt.ylabel('Number of Recommendations')
plt.title('Recommendations by Cluster')
plt.xticks(range(optimal_n_clusters), [f'Cluster {i}' for i in range(optimal_n_clusters)])
plt.legend()

plt.tight_layout()
plt.savefig('cluster_recommendations.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n--- Analysis Complete ---")
print("All visualizations have been saved.")
print(f"Best model: {best_model} with {optimal_n_clusters} clusters")
