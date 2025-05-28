#!/usr/bin/env python3
"""
Complete PageRank vs HITS Analysis Suite
Runs all analyses and generates comprehensive comparison report
"""

import os
import time
import networkx as nx
from graph_theory import GraphAnalyzer
from dataset_downloader import analyze_real_dataset, get_sample_datasets

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
    import networkx as nx
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
    
    # Create comprehensive summary
    summary_content = f"""
PAGERANK vs HITS ANALYSIS SUMMARY
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Analysis Time: {time.time() - start_time:.2f} seconds

DATASETS ANALYZED:
"""
    
    for i, file in enumerate(analysis_files, 1):
        dataset_name = file.replace('_analysis_report.txt', '')
        summary_content += f"{i}. {dataset_name}\n"
    
    summary_content += f"""
GENERATED FILES:
- Analysis Reports: {len(analysis_files)}
- Comparison Plots: {len(comparison_images)}
- Network Data Files: {len([f for f in os.listdir('.') if f.endswith('.txt') and 'processed' not in f and 'report' not in f])}

FILES CREATED:
"""
    
    all_files = sorted([f for f in os.listdir('.') if f.endswith(('.txt', '.png', '.md')) and not f.startswith('.')])
    for file in all_files:
        size = os.path.getsize(file)
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.1f}MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size}B"
        summary_content += f"- {file} ({size_str})\n"
    
    # Save summary
    with open("ANALYSIS_SUMMARY.txt", "w") as f:
        f.write(summary_content)
    
    print(f"\n4. ANALYSIS COMPLETE!")
    print("-" * 40)
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Files generated: {len(all_files)}")
    print(f"Reports created: {len(analysis_files)}")
    print(f"Visualizations: {len(comparison_images)}")
    print("\nKey files:")
    print("- FINAL_COMPARISON_REPORT.md (comprehensive report)")
    print("- ANALYSIS_SUMMARY.txt (execution summary)")
    print("- *_analysis_report.txt (individual dataset reports)")
    print("- *_comparison.png (visualization plots)")
    
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
            print("1. Review FINAL_COMPARISON_REPORT.md for comprehensive results")
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
