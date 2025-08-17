import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#pip install torch torchvision fastai pandas matplotlib seaborn
import torch
from fastai.vision.all import *
from pathlib import Path
import time
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

def setup_device():
    """Setup the compute device prioritizing MPS (Apple Silicon)"""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def verify_mps_setup():
    """Verify MPS setup and print relevant information"""
    print("\nSystem Configuration:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS backend built: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS fallback enabled: {os.getenv('PYTORCH_ENABLE_MPS_FALLBACK', '0')}")

def setup_output_directory():
    """Setup output directory with iteration tracking"""
    # Create base output directory
    output_base = Path('model_test_results')
    output_base.mkdir(exist_ok=True)
    
    # Find existing iteration folders
    existing_iterations = [f.name for f in output_base.iterdir() if f.is_dir() and f.name.startswith('iteration_')]
    
    # Get next iteration number
    if not existing_iterations:
        next_iteration = 1
    else:
        iteration_numbers = [int(name.split('_')[1]) for name in existing_iterations]
        next_iteration = max(iteration_numbers) + 1
    
    # Create new iteration directory
    iteration_dir = output_base / f'iteration_{next_iteration}'
    iteration_dir.mkdir(exist_ok=True)
    
    return iteration_dir

def load_model(model_path, device):
    """Load the exported FastAI model and move to appropriate device"""
    try:
        learn = load_learner(model_path)
        learn.dls.to(device)  # Move DataLoaders to device
        print(f"Successfully loaded model from {model_path} to {device}")
        return learn
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_batch(learn, batch_paths, device):
    """Make predictions for a batch of images"""
    results = []
    for img_path in batch_paths:
        try:
            img = PILImage.create(img_path)
            pred_class, pred_idx, probs = learn.predict(img)
            results.append({
                'file_name': img_path.name,
                'prediction': pred_class,
                'confidence': float(probs[pred_idx])
            })
        except Exception as e:
            print(f"Error predicting {img_path.name}: {e}")
    return results
def save_test_results(results, iteration_dir, process_time, total_images):
    """Save test results in multiple formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed CSV
    csv_path = iteration_dir / f'predictions_{timestamp}.csv'
    pd.DataFrame(results).to_csv(csv_path, index=False)
    
    # Calculate statistics
    df = pd.DataFrame(results)
    class_distribution = df['prediction'].value_counts().to_dict()
    
    # Create summary statistics
    summary_stats = {
        'timestamp': timestamp,
        'total_images': total_images,
        'total_processing_time': round(process_time, 2),
        'images_per_second': round(total_images/process_time, 2),
        'average_confidence': round(float(df['confidence'].mean()), 4),
        'class_distribution': class_distribution,
        'system_info': {
            'pytorch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built(),
            'cuda_available': torch.cuda.is_available()
        }
    }
    
    # Save JSON summary
    json_path = iteration_dir / f'summary_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    # Create detailed text report
    report_path = iteration_dir / f'report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write("Model Test Results\n")
        f.write("=================\n\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Total Images Processed: {total_images}\n")
        f.write(f"Total Processing Time: {process_time:.2f} seconds\n")
        f.write(f"Images per Second: {total_images/process_time:.2f}\n")
        f.write(f"Average Confidence: {df['confidence'].mean():.4f}\n\n")
        
        f.write("Class Distribution:\n")
        for class_name, count in class_distribution.items():
            percentage = (count / total_images) * 100
            f.write(f"{class_name}: {count} images ({percentage:.1f}%)\n")
        
        f.write("\nConfidence Distribution:\n")
        f.write(f"Min Confidence: {df['confidence'].min():.4f}\n")
        f.write(f"Max Confidence: {df['confidence'].max():.4f}\n")
        f.write(f"Median Confidence: {df['confidence'].median():.4f}\n\n")
        
        f.write("System Information:\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"MPS Available: {torch.backends.mps.is_available()}\n")
        f.write(f"MPS Built: {torch.backends.mps.is_built()}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
    
    return csv_path, json_path, report_path

def load_previous_iterations(output_base):
    """Load and parse all previous iteration results"""
    iterations = {}
    
    for iter_dir in sorted(output_base.glob('iteration_*')):
        iter_num = int(iter_dir.name.split('_')[1])
        
        # Find the most recent summary file in this iteration
        summary_files = list(iter_dir.glob('summary_*.json'))
        if not summary_files:
            continue
            
        latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_summary, 'r') as f:
            iterations[iter_num] = json.load(f)
    
    return iterations
def generate_comparison_report(iterations, current_iteration, output_dir):
    """Generate comparative analysis between iterations"""
    if not iterations:
        print("No previous iterations found for comparison.")
        return None, None
    
    # Create DataFrame for easy analysis
    comparison_data = []
    for iter_num, data in iterations.items():
        row = {
            'iteration': iter_num,
            'total_images': data['total_images'],
            'processing_time': data['total_processing_time'],
            'images_per_second': data['images_per_second'],
            'average_confidence': data['average_confidence'],
        }
        # Add class distribution
        for class_name, count in data['class_distribution'].items():
            row[f'{class_name}_count'] = count
            row[f'{class_name}_percentage'] = (count / data['total_images']) * 100
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Generate comparison plots
    plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance trends
    axes[0, 0].plot(df_comparison['iteration'], df_comparison['images_per_second'], 
                    marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_title('Processing Speed Trend')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Images per Second')
    
    # Confidence trend
    axes[0, 1].plot(df_comparison['iteration'], df_comparison['average_confidence'], 
                    marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_title('Average Confidence Trend')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Average Confidence')
    
    # Class distribution trend
    class_columns = [col for col in df_comparison.columns if col.endswith('_percentage')]
    class_data = df_comparison[class_columns]
    
    # Stacked bar chart for class distribution
    class_data.plot(kind='bar', stacked=True, ax=axes[1, 0])
    axes[1, 0].set_title('Class Distribution Across Iterations')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Percentage')
    
    # Processing time comparison
    axes[1, 1].bar(df_comparison['iteration'], df_comparison['processing_time'])
    axes[1, 1].set_title('Processing Time Comparison')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Seconds')
    
    plt.tight_layout()
    
    # Save comparison plots
    comparison_plot_path = output_dir / f'iteration_comparison_{current_iteration}.png'
    plt.savefig(comparison_plot_path)
    plt.close()
    
    # Generate comparison report
    report_path = output_dir / f'iteration_comparison_report_{current_iteration}.txt'
    with open(report_path, 'w') as f:
        f.write("Iteration Comparison Report\n")
        f.write("=========================\n\n")
        
        f.write("Performance Trends:\n")
        f.write("-----------------\n")
        perf_trends = df_comparison[['iteration', 'images_per_second', 'processing_time', 'average_confidence']]
        f.write(f"{perf_trends.to_string()}\n\n")
        
        # Calculate changes from previous iteration
        if len(df_comparison) > 1:
            current = df_comparison.iloc[-1]
            previous = df_comparison.iloc[-2]
            
            f.write("Changes from Previous Iteration:\n")
            f.write("------------------------------\n")
            f.write(f"Speed Change: {(current['images_per_second'] - previous['images_per_second']):.2f} images/sec\n")
            f.write(f"Confidence Change: {(current['average_confidence'] - previous['average_confidence']):.4f}\n")
            
            # Class distribution changes
            f.write("\nClass Distribution Changes:\n")
            for class_col in class_columns:
                class_name = class_col.replace('_percentage', '')
                change = current[class_col] - previous[class_col]
                f.write(f"{class_name}: {change:+.1f}%\n")
        
        # Add summary statistics
        f.write("\nSummary Statistics:\n")
        f.write("-----------------\n")
        f.write(f"Total Iterations: {len(df_comparison)}\n")
        f.write(f"Best Performance: {df_comparison['images_per_second'].max():.2f} images/sec (Iteration {df_comparison['images_per_second'].idxmax()})\n")
        f.write(f"Highest Confidence: {df_comparison['average_confidence'].max():.4f} (Iteration {df_comparison['average_confidence'].idxmax()})\n")
    
    return comparison_plot_path, report_path
def process_test_folder(model_path, test_folder, batch_size=32):
    """Process all images in the test folder using GPU acceleration"""
    # Setup output directory
    output_base = Path('model_test_results')
    iteration_dir = setup_output_directory()
    print(f"\nResults will be saved in: {iteration_dir}")
    
    # Verify MPS setup
    verify_mps_setup()
    
    # Setup device
    device = setup_device()
    
    # Load the model
    learn = load_model(model_path, device)
    if learn is None:
        return None, None
    
    # Convert paths to Path objects
    test_path = Path(test_folder)
    
    # Verify test folder exists
    if not test_path.exists():
        print(f"Error: Test folder {test_path} does not exist!")
        return None, None
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in test_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {test_path}")
        return None, None
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Process images in batches
    results = []
    start_time = time.time()
    
    # Create batches
    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        # Process batch
        batch_results = predict_batch(learn, batch_paths, device)
        results.extend(batch_results)
        
        # Show progress
        processed = min(i + batch_size, len(image_files))
        print(f"Processed {processed}/{len(image_files)} images")
    
    # Calculate processing time
    end_time = time.time()
    process_time = end_time - start_time
    
    # Create DataFrame for current results
    df = pd.DataFrame(results)
    
    # Save test results
    csv_path, json_path, report_path = save_test_results(
        results,
        iteration_dir,
        process_time,
        len(results)
    )
    
    # Load previous iterations and generate comparison
    iterations = load_previous_iterations(output_base)
    current_iteration = int(iteration_dir.name.split('_')[1])
    
    # Add current iteration to the comparison
    iterations[current_iteration] = {
        'total_images': len(results),
        'total_processing_time': process_time,
        'images_per_second': len(results)/process_time,
        'average_confidence': float(df['confidence'].mean()),
        'class_distribution': df['prediction'].value_counts().to_dict()
    }
    
    # Generate comparison report
    comparison_plot_path, comparison_report_path = generate_comparison_report(
        iterations,
        current_iteration,
        iteration_dir
    )
    
    print("\nResults saved to:")
    print(f"CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Detailed Report: {report_path}")
    if comparison_plot_path:
        print(f"Comparison Plot: {comparison_plot_path}")
    if comparison_report_path:
        print(f"Comparison Report: {comparison_report_path}")
    
    return results, iteration_dir

if __name__ == "__main__":
    # Define paths


    '''_______________________'''
    
    
    model_path = 'export.pkl'  # Path to your exported model

    '''_________________________'''
    test_folder = '/Users/travisyounker/Documents/Side Projects/Eden Intelligence./No Commercials/small_commmercial_data/TEST'
    
    # Run the inference
    results, output_dir = process_test_folder(model_path, test_folder)
    
    if results is None:
        print("\nProcessing failed. Please check the error messages above.")
    else:
        print(f"\nProcessing completed successfully. Results are saved in: {output_dir}")