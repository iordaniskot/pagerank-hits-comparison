#!/usr/bin/env python3
"""
Standalone cleanup utility for PageRank vs HITS project
Removes temporary files created during dataset analysis

Usage:
    python cleanup_utility.py [--all | --dataset DATASET_NAME]
"""

import os
import argparse
import glob
from dataset_downloader import cleanup_temporary_files

def main():
    """
    Main function for standalone cleanup utility
    """
    parser = argparse.ArgumentParser(
        description="Clean up temporary files from PageRank vs HITS analysis"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Clean up all temporary files in the current directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Clean up files for a specific dataset only"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List temporary files without removing them"
    )
    
    args = parser.parse_args()
    
    print("PageRank vs HITS Cleanup Utility")
    print("=" * 40)
    
    if args.list:
        # List temporary files without removing
        temp_patterns = ["*.mtx", "*.edges", "*.zip", "*_processed.txt"]
        found_files = []
        
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                if not any(keep_pattern in file_path for keep_pattern in 
                          ['_analysis_report', '_comparison', 'ANALYSIS_SUMMARY']):
                    found_files.append(file_path)
        
        if found_files:
            print("Temporary files found:")
            for file in sorted(found_files):
                size = os.path.getsize(file)
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"  - {file} ({size_str})")
        else:
            print("No temporary files found.")
        return
    
    if args.all:
        print("Cleaning up all temporary files...")
        cleanup_temporary_files()
    elif args.dataset:
        print(f"Cleaning up files for dataset: {args.dataset}")
        cleanup_temporary_files(args.dataset)
    else:
        # Interactive mode
        print("Select cleanup option:")
        print("1. Clean all temporary files")
        print("2. Clean files for specific dataset")
        print("3. List temporary files only")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nCleaning up all temporary files...")
            cleanup_temporary_files()
        elif choice == "2":
            dataset_name = input("Enter dataset name: ").strip()
            if dataset_name:
                print(f"\nCleaning up files for dataset: {dataset_name}")
                cleanup_temporary_files(dataset_name)
            else:
                print("No dataset name provided.")
        elif choice == "3":
            # List files
            temp_patterns = ["*.mtx", "*.edges", "*.zip", "*_processed.txt"]
            found_files = []
            
            for pattern in temp_patterns:
                for file_path in glob.glob(pattern):
                    if not any(keep_pattern in file_path for keep_pattern in 
                              ['_analysis_report', '_comparison', 'ANALYSIS_SUMMARY']):
                        found_files.append(file_path)
            
            if found_files:
                print("\nTemporary files found:")
                for file in sorted(found_files):
                    size = os.path.getsize(file)
                    if size > 1024*1024:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    elif size > 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    print(f"  - {file} ({size_str})")
            else:
                print("\nNo temporary files found.")
        elif choice == "4":
            print("Exiting...")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
