#!/usr/bin/env python3
"""
Complete PageRank vs HITS Analysis Suite
Runs all analyses and generates comprehensive comparison report
"""

import os
import time
import networkx as nx
from graph_theory import GraphAnalyzer
from dataset_downloader import analyze_real_dataset, get_sample_datasets, cleanup_temporary_files

def run_complete_analysis():
    """
    Run complete analysis suite including synthetic and real datasets
    """
    print("="*80)
    print("PAGERANK vs HITS COMPLETE ANALYSIS SUITE")
    print("="*80)
    
    start_time = time.time()
    
    # Initialize analyzer
    analyzer = GraphAnalyzer()
    
    print("\n1. SYNTHETIC DATASET ANALYSIS")
    print("-" * 40)
    
    # Test with sample graph
    print("Creating sample graph...")
    sample_graph = analyzer.create_sample_graph()
    analyzer.graph = sample_graph  # Set the current graph
    
    print("Running analysis on sample graph...")
    results = analyzer.compare_with_networkx()
    correlations = analyzer.analyze_correlations(results)
    report = analyzer.generate_performance_report(results, correlations)
    
    # Save sample results
    with open("sample_analysis_report.txt", "w") as f:
        f.write("SAMPLE GRAPH ANALYSIS REPORT\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    analyzer.plot_comparison(results, "sample_analysis_comparison.png")
    print("✓ Sample graph analysis complete")
    
    # Test with larger synthetic graph using NetworkX
    print("\nCreating larger synthetic graph...")
    large_graph = nx.erdos_renyi_graph(100, 0.06, directed=True)
    analyzer.graph = large_graph
    
    print("Running analysis on large synthetic graph...")
    results = analyzer.compare_with_networkx()
    correlations = analyzer.analyze_correlations(results)
    report = analyzer.generate_performance_report(results, correlations)
    
    # Save large graph results
    with open("large_synthetic_analysis_report.txt", "w") as f:
        f.write("LARGE SYNTHETIC GRAPH ANALYSIS REPORT\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    analyzer.plot_comparison(results, "large_synthetic_comparison.png")
    print("✓ Large synthetic graph analysis complete")
    
    print("\n2. REAL-WORLD DATASET ANALYSIS")
    print("-" * 40)
    
    # Analyze all available real datasets
    datasets = get_sample_datasets()
    
    for dataset_name in datasets.keys():
        print(f"\nAnalyzing {dataset_name}...")
        try:
            analyze_real_dataset(dataset_name)
            print(f"✓ {dataset_name} analysis complete")
        except Exception as e:
            print(f"✗ Error analyzing {dataset_name}: {e}")
    
    print("\n3. GENERATING SUMMARY STATISTICS")
    print("-" * 40)
    
    # Collect all analysis files
    analysis_files = []
    comparison_images = []
    
    for file in os.listdir('.'):
        if file.endswith('_analysis_report.txt'):
            analysis_files.append(file)
        elif file.endswith('_comparison.png'):
            comparison_images.append(file)
    
   
    
    for i, file in enumerate(analysis_files, 1):
        dataset_name = file.replace('_analysis_report.txt', '')

    
   
    
    all_files = sorted([f for f in os.listdir('.') if f.endswith(('.txt', '.png', '.md')) and not f.startswith('.')])


    
    print(f"\n4. FINAL CLEANUP")
    print("-" * 40)
    
    # Perform final cleanup of any remaining temporary files
    print("Cleaning up any remaining temporary files...")
    cleanup_temporary_files()
    
    print(f"\n5. ANALYSIS COMPLETE!")
    print("-" * 40)
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Files generated: {len(all_files)}")
    print(f"Reports created: {len(analysis_files)}")
    print(f"Visualizations: {len(comparison_images)}")
    print("\nKey files:")
    print("- *_analysis_report.txt (individual dataset reports)")
    print("- *_comparison.png (visualization plots)")
    print("\n✓ All temporary files automatically cleaned up!")
    
    return True

def main():
    """
    Main function to run complete analysis
    """
    print("Starting complete PageRank vs HITS analysis...")
    print("This will analyze synthetic and real-world datasets")
    print("Estimated time: 2-5 minutes depending on network speed")
    
    try:
        success = run_complete_analysis()
        if success:
            print("\n🎉 Analysis suite completed successfully!")
            print("\nNext steps:")
            print("2. Check individual dataset reports for detailed analysis")
            print("3. View comparison plots for visual insights")
        else:
            print("\n❌ Analysis suite encountered errors")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
