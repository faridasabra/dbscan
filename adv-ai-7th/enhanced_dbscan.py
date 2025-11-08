import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory for images
output_dir = "enhanced_dbscan_results"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("ENHANCED DBSCAN IMPLEMENTATION")
print("AAST - Advanced Artificial Intelligence - CA15105")
print("=" * 70)

class EnhancedDBSCAN:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def generate_datasets(self):
        """Generate spherical and non-spherical datasets"""
        np.random.seed(42)
        
        # Spherical dataset
        spherical_data, _ = make_blobs(n_samples=1000, centers=4, 
                                     cluster_std=0.6, random_state=42)
        
        # Non-spherical dataset (moons)
        non_spherical_data, _ = make_moons(n_samples=1000, noise=0.1, 
                                         random_state=42)
        
        return spherical_data, non_spherical_data
    
    def disk_kmeans(self, data, k=4):
        """Apply K-Means clustering for partitioning"""
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        return labels, kmeans.cluster_centers_
    
    def apply_dbscan_partition(self, data, eps=0.3, min_samples=5):
        """Apply DBSCAN to a single partition"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        # Get core points
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        
        return labels, core_samples_mask, dbscan
    
    def enhanced_dbscan_partitioned(self, data, k=4, eps=0.3, min_samples=5):
        """Enhanced DBSCAN on partitioned dataset"""
        # Step 1: Partition using K-Means
        kmeans_labels, centers = self.disk_kmeans(data, k)
        
        # Step 2: Apply DBSCAN to each partition
        all_dbscan_labels = np.full(len(data), -1)
        all_core_points = np.zeros(len(data), dtype=bool)
        
        current_label = 0
        for cluster_id in range(k):
            # Get data points in current partition
            partition_mask = kmeans_labels == cluster_id
            partition_data = data[partition_mask]
            
            if len(partition_data) > 0:
                # Apply DBSCAN to partition
                dbscan_labels, core_mask, _ = self.apply_dbscan_partition(
                    partition_data, eps, min_samples)
                
                # Adjust labels to be unique across partitions
                adjusted_labels = np.where(dbscan_labels != -1, 
                                         dbscan_labels + current_label, -1)
                all_dbscan_labels[partition_mask] = adjusted_labels
                all_core_points[partition_mask] = core_mask
                
                # Update current_label for next partition
                if len(dbscan_labels[dbscan_labels != -1]) > 0:
                    current_label = np.max(adjusted_labels[adjusted_labels != -1]) + 1
        
        return kmeans_labels, all_dbscan_labels, all_core_points, centers
    
    def dbscan_data_reduction(self, data, eps=0.3, min_samples=5):
        """Apply DBSCAN for data reduction and border detection"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        # Get core points, border points, and noise
        core_points_mask = np.zeros_like(labels, dtype=bool)
        core_points_mask[dbscan.core_sample_indices_] = True
        
        border_points_mask = np.zeros_like(labels, dtype=bool)
        noise_mask = labels == -1
        
        # Identify border points (points that are not core but not noise)
        for i in range(len(data)):
            if not core_points_mask[i] and not noise_mask[i]:
                border_points_mask[i] = True
        
        return labels, core_points_mask, border_points_mask, noise_mask, dbscan
    
    def calculate_reduction_ratio(self, border_points_mask, total_points):
        """Calculate data reduction ratio"""
        border_points_count = np.sum(border_points_mask)
        return border_points_count / total_points
    
    def plot_task1_results(self, data, kmeans_labels, dbscan_labels, core_points, centers, title, save_path=None):
        """Plot results for Task 1 and save to file"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Enhanced DBSCAN - {title}', fontsize=16, fontweight='bold')
        
        # 1. Original dataset
        axes[0, 0].scatter(data[:, 0], data[:, 1], s=30, alpha=0.7, c='blue')
        axes[0, 0].set_title('1. Original Dataset')
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. K-Means partitions
        unique_kmeans = np.unique(kmeans_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_kmeans)))
        
        for i, cluster_id in enumerate(unique_kmeans):
            cluster_data = data[kmeans_labels == cluster_id]
            axes[0, 1].scatter(cluster_data[:, 0], cluster_data[:, 1], 
                             s=30, alpha=0.7, c=[colors[i]], label=f'Partition {cluster_id}')
        
        axes[0, 1].scatter(centers[:, 0], centers[:, 1], marker='x', s=200, 
                         linewidths=3, color='red', label='Centroids')
        axes[0, 1].set_title('2. K-Means Partitions')
        axes[0, 1].set_xlabel('Feature 1')
        axes[0, 1].set_ylabel('Feature 2')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. DBSCAN results on partitions
        unique_dbscan = np.unique(dbscan_labels)
        colors_dbscan = plt.cm.tab10(np.linspace(0, 1, len(unique_dbscan)))
        
        for i, cluster_id in enumerate(unique_dbscan):
            if cluster_id == -1:
                # Noise points
                noise_data = data[dbscan_labels == cluster_id]
                axes[1, 0].scatter(noise_data[:, 0], noise_data[:, 1], 
                                 s=30, alpha=0.7, c='gray', label='Noise')
            else:
                cluster_data = data[dbscan_labels == cluster_id]
                axes[1, 0].scatter(cluster_data[:, 0], cluster_data[:, 1], 
                                 s=30, alpha=0.7, c=[colors_dbscan[i]], 
                                 label=f'Cluster {cluster_id}')
        
        axes[1, 0].set_title('3. DBSCAN Clusters (All Points)')
        axes[1, 0].set_xlabel('Feature 1')
        axes[1, 0].set_ylabel('Feature 2')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Core points only (dense regions)
        core_data = data[core_points]
        non_core_data = data[~core_points]
        
        axes[1, 1].scatter(non_core_data[:, 0], non_core_data[:, 1], 
                         s=30, alpha=0.3, c='lightgray', label='Non-core')
        axes[1, 1].scatter(core_data[:, 0], core_data[:, 1], 
                         s=50, alpha=0.8, c='red', label='Core Points')
        axes[1, 1].set_title('4. Dense Regions (Core Points Only)')
        axes[1, 1].set_xlabel('Feature 1')
        axes[1, 1].set_ylabel('Feature 2')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()
    
    def plot_task2_results(self, data, labels, core_points, border_points, noise_points, title, save_path=None):
        """Plot results for Task 2 and save to file"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'DBSCAN for Data Reduction - {title}', fontsize=16, fontweight='bold')
        
        # 1. Raw dataset
        axes[0].scatter(data[:, 0], data[:, 1], s=30, alpha=0.7, c='blue')
        axes[0].set_title('1. Raw Dataset')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].grid(True, alpha=0.3)
        
        # 2. DBSCAN clustering results
        # Core points (colored by cluster)
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            cluster_core_mask = (labels == label) & core_points
            cluster_data = data[cluster_core_mask]
            axes[1].scatter(cluster_data[:, 0], cluster_data[:, 1], 
                          s=50, alpha=0.8, c=[colors[i]], label=f'Cluster {label} Core')
        
        # Border points
        border_data = data[border_points]
        axes[1].scatter(border_data[:, 0], border_data[:, 1], 
                      s=40, alpha=0.7, c='orange', marker='s', label='Border Points')
        
        # Noise points
        noise_data = data[noise_points]
        axes[1].scatter(noise_data[:, 0], noise_data[:, 1], 
                      s=30, alpha=0.5, c='gray', marker='x', label='Noise')
        
        axes[1].set_title('2. DBSCAN: Core, Border, and Noise')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Data reduction - border points only
        axes[2].scatter(border_data[:, 0], border_data[:, 1], 
                      s=50, alpha=0.8, c='red', marker='s')
        axes[2].set_title('3. Reduced Dataset (Border Points Only)')
        axes[2].set_xlabel('Feature 1')
        axes[2].set_ylabel('Feature 2')
        axes[2].grid(True, alpha=0.3)
        
        # Draw convex hull around border points for each cluster
        for label in unique_labels:
            if label == -1:
                continue
            cluster_border_mask = (labels == label) & border_points
            cluster_border_data = data[cluster_border_mask]
            
            if len(cluster_border_data) > 2:
                try:
                    hull = ConvexHull(cluster_border_data)
                    for simplex in hull.simplices:
                        axes[2].plot(cluster_border_data[simplex, 0], 
                                   cluster_border_data[simplex, 1], 'b-', alpha=0.6)
                except:
                    pass
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()

def main():
    # Initialize Enhanced DBSCAN
    enhanced_dbscan = EnhancedDBSCAN()
    
    # Generate datasets
    spherical_data, non_spherical_data = enhanced_dbscan.generate_datasets()
    
    print("\nTASK 1: Enhanced DBSCAN on Partitioned Dataset")
    print("-" * 50)
    
    # Parameters for Task 1
    k_partitions = 4
    eps_task1 = 0.3
    min_samples_task1 = 10
    
    # Process spherical dataset
    print("\nProcessing Spherical Dataset...")
    kmeans_labels_sph, dbscan_labels_sph, core_points_sph, centers_sph = \
        enhanced_dbscan.enhanced_dbscan_partitioned(
            spherical_data, k=k_partitions, eps=eps_task1, min_samples=min_samples_task1)
    
    enhanced_dbscan.plot_task1_results(
        spherical_data, kmeans_labels_sph, dbscan_labels_sph, 
        core_points_sph, centers_sph, "Spherical Dataset",
        save_path=f"{output_dir}/task1_spherical_results.png")
    
    # Process non-spherical dataset
    print("Processing Non-Spherical Dataset...")
    kmeans_labels_nsph, dbscan_labels_nsph, core_points_nsph, centers_nsph = \
        enhanced_dbscan.enhanced_dbscan_partitioned(
            non_spherical_data, k=k_partitions, eps=eps_task1, min_samples=min_samples_task1)
    
    enhanced_dbscan.plot_task1_results(
        non_spherical_data, kmeans_labels_nsph, dbscan_labels_nsph, 
        core_points_nsph, centers_nsph, "Non-Spherical Dataset",
        save_path=f"{output_dir}/task1_non_spherical_results.png")
    
    print("\nTASK 2: DBSCAN for Data Reduction and Border Detection")
    print("-" * 60)
    
    # Test different hyperparameter settings
    param_settings = [
        {'eps': 0.2, 'min_samples': 5},
        {'eps': 0.3, 'min_samples': 10},
        {'eps': 0.4, 'min_samples': 15}
    ]
    
    for i, params in enumerate(param_settings):
        print(f"\nParameter Setting {i+1}: eps={params['eps']}, min_samples={params['min_samples']}")
        print("-" * 40)
        
        # Process spherical dataset
        labels_sph, core_sph, border_sph, noise_sph, _ = \
            enhanced_dbscan.dbscan_data_reduction(
                spherical_data, eps=params['eps'], min_samples=params['min_samples'])
        
        reduction_ratio_sph = enhanced_dbscan.calculate_reduction_ratio(
            border_sph, len(spherical_data))
        
        print(f"Spherical Dataset - Reduction Ratio: {reduction_ratio_sph:.3f}")
        print(f"  Core points: {np.sum(core_sph)}, Border points: {np.sum(border_sph)}, Noise: {np.sum(noise_sph)}")
        
        # Process non-spherical dataset
        labels_nsph, core_nsph, border_nsph, noise_nsph, _ = \
            enhanced_dbscan.dbscan_data_reduction(
                non_spherical_data, eps=params['eps'], min_samples=params['min_samples'])
        
        reduction_ratio_nsph = enhanced_dbscan.calculate_reduction_ratio(
            border_nsph, len(non_spherical_data))
        
        print(f"Non-Spherical Dataset - Reduction Ratio: {reduction_ratio_nsph:.3f}")
        print(f"  Core points: {np.sum(core_nsph)}, Border points: {np.sum(border_nsph)}, Noise: {np.sum(noise_nsph)}")
        
        # Save plots for all parameter settings
        enhanced_dbscan.plot_task2_results(
            spherical_data, labels_sph, core_sph, border_sph, noise_sph,
            f"Spherical Dataset (eps={params['eps']}, min_samples={params['min_samples']})",
            save_path=f"{output_dir}/task2_spherical_params_{i+1}.png")
        
        enhanced_dbscan.plot_task2_results(
            non_spherical_data, labels_nsph, core_nsph, border_nsph, noise_nsph,
            f"Non-Spherical Dataset (eps={params['eps']}, min_samples={params['min_samples']})",
            save_path=f"{output_dir}/task2_non_spherical_params_{i+1}.png")
    
    # Additional analysis: Compare cluster quality
    print("\nCLUSTER QUALITY ANALYSIS:")
    print("-" * 30)
    
    # For spherical dataset with different parameters
    for i, params in enumerate(param_settings):
        labels_sph, _, _, _, _ = enhanced_dbscan.dbscan_data_reduction(
            spherical_data, eps=params['eps'], min_samples=params['min_samples'])
        
        # Calculate silhouette score (excluding noise points)
        valid_labels = labels_sph != -1
        if np.sum(valid_labels) > 1 and len(np.unique(labels_sph[valid_labels])) > 1:
            score = silhouette_score(spherical_data[valid_labels], labels_sph[valid_labels])
            print(f"Spherical Dataset - Params {i+1}: Silhouette Score = {score:.3f}")
    
    # Visualize the datasets separately for better understanding and save
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(spherical_data[:, 0], spherical_data[:, 1], s=30, alpha=0.7, c='blue')
    ax1.set_title('Spherical Dataset (Blobs)')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(non_spherical_data[:, 0], non_spherical_data[:, 1], s=30, alpha=0.7, c='red')
    ax2.set_title('Non-Spherical Dataset (Moons)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/dataset_comparison.png")
    plt.close()
    
    # Create a summary visualization of reduction ratios
    param_names = [f"eps={p['eps']}\nmin_samples={p['min_samples']}" for p in param_settings]
    reduction_ratios_sph = []
    reduction_ratios_nsph = []
    
    for params in param_settings:
        _, _, border_sph, _, _ = enhanced_dbscan.dbscan_data_reduction(
            spherical_data, eps=params['eps'], min_samples=params['min_samples'])
        reduction_ratios_sph.append(enhanced_dbscan.calculate_reduction_ratio(border_sph, len(spherical_data)))
        
        _, _, border_nsph, _, _ = enhanced_dbscan.dbscan_data_reduction(
            non_spherical_data, eps=params['eps'], min_samples=params['min_samples'])
        reduction_ratios_nsph.append(enhanced_dbscan.calculate_reduction_ratio(border_nsph, len(non_spherical_data)))
    
    # Plot reduction ratios comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(param_names))
    width = 0.35
    
    ax.bar(x - width/2, reduction_ratios_sph, width, label='Spherical Dataset', alpha=0.7)
    ax.bar(x + width/2, reduction_ratios_nsph, width, label='Non-Spherical Dataset', alpha=0.7)
    
    ax.set_xlabel('Parameter Settings')
    ax.set_ylabel('Reduction Ratio')
    ax.set_title('Data Reduction Ratios by Parameter Settings')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reduction_ratios_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/reduction_ratios_comparison.png")
    plt.close()
    
    # Final statistics
    print("\nFINAL STATISTICS:")
    print("-" * 20)
    print(f"Total points in each dataset: {len(spherical_data)}")
    print(f"Number of partitions in Task 1: {k_partitions}")
    print(f"Parameter settings tested in Task 2: {len(param_settings)}")
    print(f"All results saved to: {output_dir}/")
    
    # Calculate overall reduction effectiveness
    best_reduction_sph = min(reduction_ratios_sph)
    best_reduction_nsph = min(reduction_ratios_nsph)
    
    print(f"Best reduction ratio (spherical): {best_reduction_sph:.3f}")
    print(f"Best reduction ratio (non-spherical): {best_reduction_nsph:.3f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY:")
    print("=" * 70)
    print("1. Enhanced DBSCAN successfully partitions datasets using K-Means")
    print("2. DBSCAN effectively identifies clusters within each partition")
    print("3. Data reduction through border point retention achieves significant compression")
    print("4. Different parameter settings show varying reduction ratios")
    print("5. Border points effectively capture cluster boundaries")
    
    print("\n" + "=" * 70)
    print("ALL IMAGES SAVED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated files:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {output_dir}/{file}")

if __name__ == "__main__":
    main()