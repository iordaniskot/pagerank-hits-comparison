"""
Utility functions for downloading and processing real network datasets
from networkrepository.com for PageRank vs HITS comparison.

Usage:
    python dataset_downloader.py
"""

import requests
import zipfile
import gzip
import os
import glob
import networkx as nx
from graph_theory import GraphAnalyzer

def cleanup_temporary_files(dataset_name: str = None, target_dir: str = "."):
    """
    Clean up temporary files created during dataset analysis
    
    Args:
        dataset_name: Specific dataset name to clean up files for (optional)
        target_dir: Directory to clean up (default: current directory)
    """
    temp_patterns = [
        "*.mtx",      # Matrix Market files
        "*.edges",    # Edge list files
        "*.zip",      # Downloaded zip files
        "*_processed.txt"  # Processed SNAP files
    ]
    
    files_removed = []
    
    if dataset_name:
        # Clean up files for specific dataset
        for pattern in temp_patterns:
            if pattern.startswith("*"):
                # For general patterns, look for dataset-specific files
                if pattern == "*.mtx":
                    file_pattern = f"{dataset_name}.mtx"
                elif pattern == "*.edges":
                    file_pattern = f"{dataset_name}.edges"
                elif pattern == "*.zip":
                    file_pattern = f"{dataset_name}.zip"
                elif pattern == "*_processed.txt":
                    file_pattern = f"{dataset_name}_processed.txt"
                
                file_path = os.path.join(target_dir, file_pattern)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        files_removed.append(file_path)
                        print(f"Removed temporary file: {file_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
    else:
        # Clean up all temporary files
        for pattern in temp_patterns:
            file_pattern = os.path.join(target_dir, pattern)
            for file_path in glob.glob(file_pattern):
                # Skip files that are clearly not temporary (e.g., analysis reports)
                if any(keep_pattern in file_path for keep_pattern in 
                       ['_analysis_report', '_comparison', 'ANALYSIS_SUMMARY']):
                    continue
                    
                try:
                    os.remove(file_path)
                    files_removed.append(file_path)
                    print(f"Removed temporary file: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
    
    if files_removed:
        print(f"\n✓ Cleanup complete: {len(files_removed)} temporary files removed")
    else:
        print("\n✓ No temporary files found to clean up")
    
    return files_removed

def download_dataset(url: str, filename: str) -> str:
    """
    Download a dataset from SNAP or other sources
    
    Args:
        url: URL of the dataset
        filename: Local filename to save the dataset
        
    Returns:
        Path to the downloaded file
    """
    print(f"Downloading {filename} from {url}...")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Downloaded {filename} successfully!")
        return filename
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading {filename}: {e}")
        return None

def extract_edges_file(zip_path: str, target_dir: str = ".") -> str:
    """
    Extract edges file from downloaded zip or decompress gzip file
    
    Args:
        zip_path: Path to the zip/gz file
        target_dir: Directory to extract to
        
    Returns:
        Path to the extracted edges file
    """
    try:
        if zip_path.endswith('.gz'):
            # Handle gzip files
            output_path = zip_path[:-3]  # Remove .gz extension
            with gzip.open(zip_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Decompressed gzip file: {output_path}")
            return output_path
            
        elif zip_path.endswith('.zip'):
            # Handle zip files
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List all files in the zip
                all_files = zip_ref.namelist()
                print(f"Files in zip: {all_files}")
                
                # Look for edges file with various patterns
                edge_files = []
                
                # Try different patterns for edge files
                patterns = ['.edges', '.txt', '.tsv', '.csv', '.mtx']
                for pattern in patterns:
                    edge_files.extend([f for f in all_files if f.endswith(pattern) and not f.startswith('readme')])
                
                # Filter out readme files and prefer files with 'edge' in name
                edge_files = [f for f in edge_files if not any(word in f.lower() for word in ['readme', 'license', 'citation'])]
                
                # If we have multiple candidates, prefer ones with edge in the name, then .mtx files
                if len(edge_files) > 1:
                    preferred = [f for f in edge_files if 'edge' in f.lower()]
                    if preferred:
                        edge_files = preferred
                    else:
                        # Prefer .mtx files if no edge files found
                        mtx_files = [f for f in edge_files if f.endswith('.mtx')]
                        if mtx_files:
                            edge_files = mtx_files
                
                if edge_files:
                    edge_file = edge_files[0]  # Take the first one
                    zip_ref.extract(edge_file, target_dir)
                    extracted_path = os.path.join(target_dir, edge_file)
                    print(f"Extracted edges file: {extracted_path}")
                    return extracted_path
                else:
                    print(f"No suitable edges file found in zip. Available files: {all_files}")
                    return None
        else:
            # File is already extracted
            return zip_path
                
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return None

def process_snap_dataset(filepath: str) -> str:
    """
    Process SNAP dataset file to remove comments and create clean edge list
    
    Args:
        filepath: Path to the SNAP dataset file
        
    Returns:
        Path to the processed file
    """
    try:
        processed_path = filepath.replace('.txt', '_processed.txt')
        
        with open(filepath, 'r') as infile, open(processed_path, 'w') as outfile:
            for line in infile:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#') and not line.startswith('%'):
                    # Split and check if we have two integers
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # Validate that both parts are integers
                            int(parts[0])
                            int(parts[1])
                            outfile.write(f"{parts[0]} {parts[1]}\n")
                        except ValueError:
                            continue
        
        print(f"Processed SNAP dataset: {processed_path}")
        return processed_path
        
    except Exception as e:
        print(f"Error processing SNAP dataset {filepath}: {e}")
        return filepath  # Return original if processing fails

def get_sample_datasets():
    """
    Dictionary of sample datasets from Network Repository and other sources
    These are small to medium sized datasets suitable for testing
    """
    datasets = {
        "bio-CE-GN": {
            "url": "https://nrvis.com/download/data/bio/bio-CE-GN.zip",
            "description": "C. elegans gene network (2.2K nodes, 53.7K edges) - Biological Networks",
            "format": "edgelist",
            "compressed": True,
            "source": "networkrepository"
        },
        "bio-celegans": {
            "url": "https://nrvis.com/download/data/bio/bio-celegans.zip",
            "description": "C. elegans neural network (453 nodes, 2K edges) - Biological Networks",
            "format": "edgelist",
            "compressed": True,
            "source": "networkrepository"
        },
        "soc-karate": {
            "url": "https://nrvis.com/download/data/soc/soc-karate.zip",
            "description": "Zachary's karate club network (34 nodes, 78 edges) - Social Networks",
            "format": "edgelist",
            "compressed": True,
            "source": "networkrepository"
        },
        "ca-netscience": {
            "url": "https://nrvis.com/download/data/ca/ca-netscience.zip",
            "description": "Network science collaboration network (1.6K nodes, 2.7K edges)",
            "format": "edgelist",
            "compressed": True,
            "source": "networkrepository"
        },
        "web-polbooks": {
            "url": "https://nrvis.com/download/data/web/web-polbooks.zip",
            "description": "Political books network (105 nodes, 441 edges) - Web Networks",
            "format": "edgelist",
            "compressed": True,
            "source": "networkrepository"
        },
        "ia-email-univ": {
            "url": "https://nrvis.com/download/data/ia/ia-email-univ.zip",
            "description": "University email network (1.1K nodes, 5.4K edges) - Interaction Networks",
            "format": "edgelist",
            "compressed": True,
            "source": "networkrepository"
        }
    }
    return datasets

def analyze_real_dataset(dataset_name: str):
    """
    Download and analyze a real dataset
    
    Args:
        dataset_name: Name of the dataset to analyze
    """
    datasets = get_sample_datasets()
    
    if dataset_name not in datasets:
        print(f"Dataset {dataset_name} not found. Available datasets:")
        for name, info in datasets.items():
            print(f"  - {name}: {info['description']}")
        return
    
    dataset = datasets[dataset_name]
    
    # Download dataset
    if dataset.get('compressed', False):
        if dataset['url'].endswith('.gz'):
            downloaded_filename = f"{dataset_name}.txt.gz"
        else:
            downloaded_filename = f"{dataset_name}.zip"
    else:
        downloaded_filename = f"{dataset_name}.txt"
    
    downloaded_file = download_dataset(dataset["url"], downloaded_filename)
    
    if not downloaded_file:
        print("Failed to download dataset")
        return
    
    # Extract/decompress if needed
    if dataset.get('compressed', False):
        edges_file = extract_edges_file(downloaded_file)
        if not edges_file:
            print("Failed to extract edges file")
            return
    else:
        edges_file = downloaded_file
    
    # Process SNAP format if needed
    if 'snap.stanford.edu' in dataset['url']:
        edges_file = process_snap_dataset(edges_file)
    
    # Analyze with GraphAnalyzer
    print(f"\nAnalyzing {dataset_name}...")
    print(f"Description: {dataset['description']}")
    
    analyzer = GraphAnalyzer()
    
    try:
        # Load the graph
        graph = analyzer.load_graph_from_edgelist(edges_file, directed=True)
        
        if graph is None:
            print("Failed to load graph")
            return
        
        # Run comparison
        print("Running PageRank vs HITS comparison...")
        results = analyzer.compare_with_networkx()
        correlations = analyzer.analyze_correlations(results)
        
        # Generate report
        report = analyzer.generate_performance_report(results, correlations)
        print(report)
        
        # Save results
        report_filename = f"{dataset_name}_analysis_report.txt"
        with open(report_filename, "w") as f:
            f.write(f"ANALYSIS REPORT FOR {dataset_name}\n")
            f.write("="*50 + "\n")
            f.write(f"Description: {dataset['description']}\n")
            f.write(f"Source: {dataset['url']}\n\n")
            f.write(report)
        
        # Create visualizations
        plot_filename = f"{dataset_name}_comparison.png"
        analyzer.plot_comparison(results, plot_filename)
        
        print(f"\nAnalysis complete! Generated files:")
        print(f"- {report_filename}")
        print(f"- {plot_filename}")
        
        # Clean up temporary files for this dataset
        print(f"\nCleaning up temporary files for {dataset_name}...")
        cleanup_temporary_files(dataset_name)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Still attempt cleanup even if analysis failed
        print(f"\nAttempting cleanup after error...")
        cleanup_temporary_files(dataset_name)

def main():
    """
    Main function to demonstrate dataset downloading and analysis
    """
    print("Real Dataset Analyzer for PageRank vs HITS Comparison")
    print("="*55)
    
    datasets = get_sample_datasets()
    
  
    
    print("Examples of usage:")
    print("1. To analyze a specific dataset:")
    print("   from dataset_downloader import analyze_real_dataset")
    print("   analyze_real_dataset('ca-GrQc')")
    print()
    print("2. To analyze all datasets:")
    print("   for dataset_name in get_sample_datasets().keys():")
    print("       analyze_real_dataset(dataset_name)")
    print()
    
  
        
    print("Do you want to run :")
    print("1. Analyze a specific dataset")
    print("2. Analyze all datasets")
    choice = input("Enter your choice (1/2): ").strip()
    if choice == '1':
        while True:
                
            print("\nAvailable datasets:")
            for i, (name, info) in enumerate(datasets.items(), 1):
                print(f"{i}. {name}")
                print(f"   {info['description']}")
                print()
            
            dataset_choice = input("Enter the dataset number or name: ").strip()
            if dataset_choice.isdigit():
                dataset_choice = list(datasets.keys())[int(dataset_choice) - 1]
                response = input(f"Do you want to analyze {dataset_choice}? (yes/no): ").strip().lower()
                if response == 'yes':
                    analyze_real_dataset(dataset_choice)
                    break
                else:
                    print("Cancelled analysis.")
            elif dataset_choice in datasets:
                dataset_choice = dataset_choice
                response = input(f"Do you want to analyze {dataset_choice}? (yes/no): ").strip().lower()
                if response == 'yes':
                    analyze_real_dataset(dataset_choice)
                    break
                else:
                    print("Cancelled analysis.")
    elif choice == '2':
        print("\nAnalyzing all available datasets...")
        for dataset_name in datasets.keys():
            print(f"\nAnalyzing {dataset_name}...")
            try:
                analyze_real_dataset(dataset_name)
                print(f"✓ {dataset_name} analysis complete")
            except Exception as e:
                print(f"✗ Error analyzing {dataset_name}: {e}")
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")
        print("No datasets selected. Exiting.")

if __name__ == "__main__":
    main()
