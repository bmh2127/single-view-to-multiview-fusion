import logging
import numpy as np
import os
import tempfile
from pathlib import Path

from common import PYTHON, DatasetMixin, configure_logging, packages, compute_n_ssim, percentile_normalization
from pipelines.inference.backend import BackendMixin
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    conda_base,
    project,
    step,
)

configure_logging()


@project(name="fusemycell")
@conda_base(
    python=PYTHON,
    packages=packages("mlflow", "evidently", "pandas", "boto3", "numpy", "tifffile", "scikit-image", "matplotlib"),
)
class Monitoring(FlowSpec, DatasetMixin, BackendMixin):
    """A monitoring pipeline to monitor the performance of a hosted 3D image fusion model.

    This pipeline runs a series of tests and generates several reports using the
    data captured by the hosted model and a reference dataset. It specifically focuses on
    metrics relevant to the FuseMyCell challenge, including N-SSIM scores and image quality.
    """

    limit = Parameter(
        "samples",
        help=(
            "The maximum number of samples that will be loaded from the production "
            "datastore to run the monitoring tests and reports. The flow will load "
            "the most recent samples."
        ),
        default=50,
    )
    
    n_ssim_threshold = Parameter(
        "n-ssim-threshold",
        help="The minimum N-SSIM threshold required for the model predictions.",
        default=0.7,
    )

    @card
    @step
    def start(self):
        """Start the monitoring pipeline."""
        import pandas as pd
        from evidently import ColumnMapping

        # Load reference dataset
        self.data_loaders = self.load_dataset()
        
        # Get validation set as reference
        self.reference_data = pd.DataFrame()
        for batch in self.data_loaders['val']:
            for i in range(len(batch['file_path'])):
                self.reference_data = pd.concat([
                    self.reference_data,
                    pd.DataFrame({
                        'file_path': [batch['file_path'][i]],
                        'input_path': [batch['file_path'][i]],
                        'ground_truth_path': [batch['file_path'][i].replace('angle', 'fused')]
                    })
                ], ignore_index=True)
        
        # Load backend implementation
        self.backend_impl = self.load_backend()

        # Load production data
        self.current_data = self.backend_impl.load(self.limit)
        
        # If no production data available, create an empty DataFrame with the expected columns
        if self.current_data is None or len(self.current_data) == 0:
            self.current_data = pd.DataFrame(columns=[
                'uuid', 'input_path', 'output_path', 'n_ssim', 'ground_truth_path', 'prediction_date'
            ])
            logging.warning("No production data available.")
        
        # Some tests require labeled data
        self.current_data_labeled = self.current_data[
            self.current_data["ground_truth_path"].notna()
        ]
        
        logging.info(f"Loaded {len(self.current_data)} production samples, {len(self.current_data_labeled)} labeled")
        
        # Define column mapping for Evidently
        self.column_mapping = ColumnMapping(
            target=None,
            prediction=None,
            numerical_features=['n_ssim'],
            datetime_features=['prediction_date']
        )

        self.next(self.compute_metrics)
    
    @card
    @step
    def compute_metrics(self):
        """Compute additional metrics for the loaded data."""
        import tifffile
        import pandas as pd
        
        # Create expanded dataframes with more metrics for analysis
        reference_metrics = []
        current_metrics = []
        
        # Process reference data if needed
        if not self.reference_data.empty:
            logging.info("Computing metrics for reference data...")
            for _, row in self.reference_data.iterrows():
                try:
                    input_path = row['input_path']
                    gt_path = row['ground_truth_path']
                    
                    if os.path.exists(input_path) and os.path.exists(gt_path):
                        # Load images
                        input_img = tifffile.imread(input_path)
                        gt_img = tifffile.imread(gt_path)
                        
                        # Ensure 3D format
                        if len(input_img.shape) == 2:
                            input_img = input_img[np.newaxis, ...]
                        if len(gt_img.shape) == 2:
                            gt_img = gt_img[np.newaxis, ...]
                        
                        # Handle 4D images (with channels)
                        if len(input_img.shape) == 4:
                            input_img = input_img[..., 0]
                        if len(gt_img.shape) == 4:
                            gt_img = gt_img[..., 0]
                        
                        # Calculate metrics
                        metrics = {
                            'file_path': input_path,
                            'ground_truth_path': gt_path,
                            'input_mean': np.mean(input_img),
                            'input_std': np.std(input_img),
                            'gt_mean': np.mean(gt_img),
                            'gt_std': np.std(gt_img),
                            'input_shape_z': input_img.shape[0],
                            'input_shape_y': input_img.shape[1],
                            'input_shape_x': input_img.shape[2],
                        }
                        reference_metrics.append(metrics)
                except Exception as e:
                    logging.warning(f"Error processing reference file {row.get('file_path', 'unknown')}: {str(e)}")
        
        # Process current data
        if not self.current_data_labeled.empty:
            logging.info("Computing metrics for production data...")
            for _, row in self.current_data_labeled.iterrows():
                try:
                    input_path = row['input_path']
                    output_path = row['output_path']
                    gt_path = row['ground_truth_path']
                    
                    if os.path.exists(input_path) and os.path.exists(output_path) and os.path.exists(gt_path):
                        # Load images
                        input_img = tifffile.imread(input_path)
                        output_img = tifffile.imread(output_path)
                        gt_img = tifffile.imread(gt_path)
                        
                        # Ensure 3D format
                        if len(input_img.shape) == 2:
                            input_img = input_img[np.newaxis, ...]
                        if len(output_img.shape) == 2:
                            output_img = output_img[np.newaxis, ...]
                        if len(gt_img.shape) == 2:
                            gt_img = gt_img[np.newaxis, ...]
                            
                        # Handle 4D images (with channels)
                        if len(input_img.shape) == 4:
                            input_img = input_img[..., 0]
                        if len(output_img.shape) == 4:
                            output_img = output_img[..., 0]
                        if len(gt_img.shape) == 4:
                            gt_img = gt_img[..., 0]
                        
                        # Calculate n_ssim if not already present in the data
                        n_ssim = row.get('n_ssim')
                        if n_ssim is None:
                            n_ssim = compute_n_ssim(output_img, gt_img, input_img)
                        
                        # Calculate metrics
                        metrics = {
                            'uuid': row.get('uuid'),
                            'input_path': input_path,
                            'output_path': output_path,
                            'ground_truth_path': gt_path,
                            'prediction_date': row.get('prediction_date'),
                            'n_ssim': n_ssim,
                            'input_mean': np.mean(input_img),
                            'input_std': np.std(input_img),
                            'output_mean': np.mean(output_img),
                            'output_std': np.std(output_img),
                            'gt_mean': np.mean(gt_img),
                            'gt_std': np.std(gt_img),
                            'input_shape_z': input_img.shape[0],
                            'input_shape_y': input_img.shape[1],
                            'input_shape_x': input_img.shape[2],
                        }
                        current_metrics.append(metrics)
                except Exception as e:
                    logging.warning(f"Error processing production file {row.get('uuid', 'unknown')}: {str(e)}")
        
        # Create DataFrames with computed metrics
        self.reference_metrics_df = pd.DataFrame(reference_metrics) if reference_metrics else pd.DataFrame()
        self.current_metrics_df = pd.DataFrame(current_metrics) if current_metrics else pd.DataFrame()
        
        logging.info(f"Computed metrics for {len(self.reference_metrics_df)} reference samples and {len(self.current_metrics_df)} production samples")
        
        self.next(self.performance_analysis)
    
    @card(type="html")
    @step
    def performance_analysis(self):
        """Generate performance analysis on the N-SSIM metric."""
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # If no metrics available, display a message
        if self.current_metrics_df.empty:
            self.html = "<h2>No labeled production data available for performance analysis.</h2>"
            self.next(self.data_quality_report)
            return
        
        # Create HTML content with model performance analysis
        html_content = []
        html_content.append("<h1>FuseMyCell Model Performance Analysis</h1>")
        
        # Calculate basic statistics
        if 'n_ssim' in self.current_metrics_df.columns:
            n_ssim_values = self.current_metrics_df['n_ssim'].dropna()
            if len(n_ssim_values) > 0:
                mean_n_ssim = n_ssim_values.mean()
                min_n_ssim = n_ssim_values.min()
                max_n_ssim = n_ssim_values.max()
                std_n_ssim = n_ssim_values.std()
                
                # Add N-SSIM summary
                html_content.append("<h2>N-SSIM Score Summary</h2>")
                html_content.append("<table border='1' style='border-collapse: collapse; width: 60%;'>")
                html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
                html_content.append(f"<tr><td>Mean N-SSIM</td><td>{mean_n_ssim:.4f}</td></tr>")
                html_content.append(f"<tr><td>Min N-SSIM</td><td>{min_n_ssim:.4f}</td></tr>")
                html_content.append(f"<tr><td>Max N-SSIM</td><td>{max_n_ssim:.4f}</td></tr>")
                html_content.append(f"<tr><td>Std N-SSIM</td><td>{std_n_ssim:.4f}</td></tr>")
                html_content.append(f"<tr><td>Samples below threshold ({self.n_ssim_threshold})</td><td>{(n_ssim_values < self.n_ssim_threshold).sum()} / {len(n_ssim_values)}</td></tr>")
                html_content.append("</table>")
                
                # Create N-SSIM histogram
                plt.figure(figsize=(10, 6))
                plt.hist(n_ssim_values, bins=20, alpha=0.7, color='skyblue')
                plt.axvline(mean_n_ssim, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_n_ssim:.4f}')
                plt.axvline(self.n_ssim_threshold, color='green', linestyle='dashed', linewidth=2, label=f'Threshold: {self.n_ssim_threshold}')
                plt.xlabel('Normalized SSIM (N-SSIM)')
                plt.ylabel('Count')
                plt.title('Distribution of N-SSIM Values')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Convert plot to base64 for HTML embedding
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png')
                img_data.seek(0)
                img_base64 = base64.b64encode(img_data.read()).decode()
                plt.close()
                
                # Add histogram to HTML
                html_content.append("<h2>N-SSIM Distribution</h2>")
                html_content.append(f'<img src="data:image/png;base64,{img_base64}" style="max-width:800px">')
                
                # Plot N-SSIM over time if timestamp data is available
                if 'prediction_date' in self.current_metrics_df.columns:
                    try:
                        # Sort by date
                        time_df = self.current_metrics_df.sort_values('prediction_date')
                        
                        plt.figure(figsize=(12, 6))
                        plt.plot(range(len(time_df)), time_df['n_ssim'], 'o-', color='blue', alpha=0.7)
                        plt.axhline(self.n_ssim_threshold, color='red', linestyle='--', label=f'Threshold: {self.n_ssim_threshold}')
                        plt.xlabel('Sample Index (chronological)')
                        plt.ylabel('N-SSIM')
                        plt.title('N-SSIM Performance Over Time')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Convert plot to base64 for HTML embedding
                        img_data = io.BytesIO()
                        plt.savefig(img_data, format='png')
                        img_data.seek(0)
                        img_base64 = base64.b64encode(img_data.read()).decode()
                        plt.close()
                        
                        # Add time series plot to HTML
                        html_content.append("<h2>N-SSIM Performance Over Time</h2>")
                        html_content.append(f'<img src="data:image/png;base64,{img_base64}" style="max-width:800px">')
                    except Exception as e:
                        logging.warning(f"Error creating time series plot: {str(e)}")
        else:
            html_content.append("<h2>No N-SSIM data available.</h2>")
        
        # Combine all HTML content
        self.html = "\n".join(html_content)
        
        self.next(self.data_quality_report)

    @card(type="html")
    @step
    def data_quality_report(self):
        """Generate a report about the quality of the data and any data drift.

        This report analyzes statistical properties of the image metadata and metrics.
        """
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
        
        # If no metrics available, display a message
        if self.current_metrics_df.empty or self.reference_metrics_df.empty:
            self.html = "<h2>Insufficient data for data quality report.</h2>"
            self.next(self.n_ssim_test)
            return
        
        # Select common features for comparison
        common_features = ['input_mean', 'input_std', 'gt_mean', 'gt_std', 
                          'input_shape_z', 'input_shape_y', 'input_shape_x']
        
        # Filter for common columns
        ref_cols = set(self.reference_metrics_df.columns)
        curr_cols = set(self.current_metrics_df.columns)
        common_features = [col for col in common_features if col in ref_cols and col in curr_cols]
        
        if not common_features:
            self.html = "<h2>No common features available for data quality report.</h2>"
            self.next(self.n_ssim_test)
            return
        
        # Create column mapping for the available features
        column_mapping = ColumnMapping(
            numerical_features=common_features
        )
        
        # Create the report
        report = Report(
            metrics=[
                DataQualityPreset(),
                DataDriftPreset(drift_share=0.2),  # Allow up to 20% of columns to drift
            ],
        )
        
        try:
            # Run the report
            report.run(
                reference_data=self.reference_metrics_df[common_features],
                current_data=self.current_metrics_df[common_features],
                column_mapping=column_mapping,
            )
            
            self.html = report.get_html()
        except Exception as e:
            self.html = f"<h2>Error generating data quality report:</h2><p>{str(e)}</p>"
            logging.exception("Error generating data quality report")
        
        self.next(self.n_ssim_test)

    @card(type="html")
    @step
    def n_ssim_test(self):
        """Run a test to check the N-SSIM scores of the model."""
        from evidently.test_suite import TestSuite
        from evidently.tests import (
            TestColumnValueMin,
            TestColumnValueMean,
        )
        
        # If no metrics available, display a message
        if self.current_metrics_df.empty:
            self.html = "<h2>No labeled production data available for N-SSIM tests.</h2>"
            self.next(self.visualize_samples)
            return
            
        # Only run if n_ssim is available
        if 'n_ssim' not in self.current_metrics_df.columns:
            self.html = "<h2>N-SSIM data not available in production data.</h2>"
            self.next(self.visualize_samples)
            return
        
        # Create the test suite for N-SSIM
        test_suite = TestSuite(
            tests=[
                # Test that mean N-SSIM is above the threshold
                TestColumnValueMean(
                    column_name="n_ssim",
                    gte=self.n_ssim_threshold,
                ),
                # Test that minimum N-SSIM is not too low
                TestColumnValueMin(
                    column_name="n_ssim",
                    gte=0.2,  # At least some improvement over input
                ),
            ],
        )

        try:
            # Run the test suite
            test_suite.run(
                current_data=self.current_metrics_df,
            )
            
            self.html = test_suite.get_html()
        except Exception as e:
            self.html = f"<h2>Error running N-SSIM tests:</h2><p>{str(e)}</p>"
            logging.exception("Error running N-SSIM tests")
        
        self.next(self.visualize_samples)

    @card(type="html")
    @step
    def visualize_samples(self):
        """Visualize sample predictions for visual inspection."""
        import tifffile
        import matplotlib.pyplot as plt
        import io
        import base64
        import random
        
        # If no labeled data available, display a message
        if self.current_metrics_df.empty:
            self.html = "<h2>No labeled production data available for visualization.</h2>"
            self.next(self.end)
            return
        
        # Start HTML content
        html_content = []
        html_content.append("<h1>Sample Predictions</h1>")
        
        # Get samples for visualization
        samples = self.current_metrics_df
        
        # Sort by N-SSIM for interesting samples
        if 'n_ssim' in samples.columns:
            # Get best, worst, and random samples
            best_samples = samples.nlargest(2, 'n_ssim')
            worst_samples = samples.nsmallest(2, 'n_ssim')
            
            # For random, exclude best and worst
            exclude_indices = best_samples.index.tolist() + worst_samples.index.tolist()
            random_candidates = samples.drop(exclude_indices) if len(exclude_indices) < len(samples) else samples
            random_samples = random_candidates.sample(min(2, len(random_candidates)))
            
            # Combine all samples
            viz_samples = pd.concat([best_samples, worst_samples, random_samples])
        else:
            # If no N-SSIM, just get random samples
            viz_samples = samples.sample(min(3, len(samples)))
        
        # Visualize each sample
        for idx, row in viz_samples.iterrows():
            try:
                input_path = row['input_path']
                output_path = row['output_path']
                gt_path = row['ground_truth_path']
                n_ssim = row.get('n_ssim', 'N/A')
                
                if os.path.exists(input_path) and os.path.exists(output_path) and os.path.exists(gt_path):
                    # Load images
                    input_img = tifffile.imread(input_path)
                    output_img = tifffile.imread(output_path)
                    gt_img = tifffile.imread(gt_path)
                    
                    # Ensure 3D format
                    if len(input_img.shape) == 2:
                        input_img = input_img[np.newaxis, ...]
                    if len(output_img.shape) == 2:
                        output_img = output_img[np.newaxis, ...]
                    if len(gt_img.shape) == 2:
                        gt_img = gt_img[np.newaxis, ...]
                        
                    # Handle 4D images (with channels)
                    if len(input_img.shape) == 4:
                        input_img = input_img[..., 0]
                    if len(output_img.shape) == 4:
                        output_img = output_img[..., 0]
                    if len(gt_img.shape) == 4:
                        gt_img = gt_img[..., 0]
                    
                    # Get middle slice for visualization
                    z_mid = input_img.shape[0] // 2
                    
                    # Normalize for visualization
                    input_slice = percentile_normalization(input_img[z_mid])
                    output_slice = percentile_normalization(output_img[z_mid])
                    gt_slice = percentile_normalization(gt_img[z_mid])
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Plot slices
                    axes[0].imshow(input_slice, cmap='gray')
                    axes[0].set_title('Input (Single View)')
                    axes[0].axis('off')
                    
                    axes[1].imshow(output_slice, cmap='gray')
                    axes[1].set_title('Prediction (Fused)')
                    axes[1].axis('off')
                    
                    axes[2].imshow(gt_slice, cmap='gray')
                    axes[2].set_title('Ground Truth (Fused)')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Convert plot to base64 for HTML embedding
                    img_data = io.BytesIO()
                    plt.savefig(img_data, format='png')
                    img_data.seek(0)
                    img_base64 = base64.b64encode(img_data.read()).decode()
                    plt.close()
                    
                    # Add to HTML
                    sample_name = os.path.basename(input_path)
                    html_content.append(f"<h2>Sample: {sample_name}</h2>")
                    html_content.append(f"<p><strong>N-SSIM:</strong> {n_ssim}</p>")
                    html_content.append(f'<img src="data:image/png;base64,{img_base64}" style="max-width:900px">')
            except Exception as e:
                logging.warning(f"Error visualizing sample {row.get('uuid', 'unknown')}: {str(e)}")
        
        if len(html_content) <= 1:
            html_content.append("<p>Could not visualize any samples. Check file paths and data formats.</p>")
        
        # Combine all HTML content
        self.html = "\n".join(html_content)
        
        self.next(self.end)

    @step
    def end(self):
        """Finish the monitoring flow."""
        logging.info("Finished monitoring flow for FuseMyCell.")

    def _message(self, message):
        """Display a message in the HTML card associated to a step."""
        self.html = message
        logging.info(message)


if __name__ == "__main__":
    Monitoring()