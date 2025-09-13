import os
import sys
import argparse
import numpy as np
from pathlib import Path
from subspace_aware_gnn import process_mat_file

def main():
    parser = argparse.ArgumentParser(description='Run Subspace-Aware Graph Neural Network for Missing Value Imputation')
    # Required arguments
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the input .mat file')

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='data/datasets', 
                        help='Output directory, defaults to data/datasets')
    parser.add_argument('--treat_zeros', action='store_true', 
                        help='Treat zero values as missing values')
    parser.add_argument('--k', type=int, default=10, 
                        help='Number of neighbors in the KNN graph')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of training epochs')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Weight coefficient for intra-subspace connections (0-1), controlling the balance between feature similarity (alpha) and label similarity (1-alpha).')
    parser.add_argument('--auto_cluster', action='store_true',
                        help='Automatically run subspace clustering after imputation')
    parser.add_argument('--list_datasets', action='store_true',
                        help='List all available .mat files in the data/datasets directory')

    args = parser.parse_args()

    # List available datasets
    if args.list_datasets:
        print("\nAvailable dataset files:")
        dataset_dir = Path("data/datasets")
        if dataset_dir.exists():
            for file_path in dataset_dir.glob("**/*.mat"):
                print(f"  {file_path}")
        else:
            print(f"  Directory {dataset_dir} does not exist")
        return

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' does not exist")
        print("\nUsage examples:")
        print("  # Data imputation only:")
        print("  python run_subspace_gnn.py --input data/datasets/COIL100_random_zero.mat --treat_zeros")
        print("  # Data imputation + automatic subspace clustering:")
        print("  python run_subspace_gnn.py --input data/datasets/COIL100_random_zero.mat --treat_zeros --auto_cluster")
        print("  # Advanced parameter settings:")
        print("  python run_subspace_gnn.py --input data/datasets/ORL_32x32_random_zero.mat --k 15 --epochs 300 --alpha 0.7 --auto_cluster")
        print("  # List available datasets:")
        print("  python run_subspace_gnn.py --list_datasets")
        return

    # Get the base name of the input file
    input_filename = os.path.basename(args.input)

    # Add check for specific datasets
    if input_filename in ["ORL_32x32.mat", "COIL100.mat"]:
        print(f"Note: Dataset '{input_filename}' is considered an original dataset with no missing data, no imputation is needed.")
        print("Please provide a version with missing values, e.g., 'COIL100_zeroed_20.mat' or 'ORL_32x32_zeroed_30.mat'.")
        return # Exit the program, do not proceed with imputation

    # Process data
    try:
        output_path = process_mat_file(
            args.input, 
            args.output_dir, 
            treat_zeros_as_missing=args.treat_zeros,
            k=args.k,
            epochs=args.epochs,
            alpha=args.alpha
        )
        
        print("\n✅ Data imputation completed!")
        print(f"📂 Output file: {output_path}")

        # If auto-clustering is enabled, run subspace clustering
        if args.auto_cluster:
            print("\n" + "="*60)
            print("🚀 Starting automatic subspace clustering...")
            print("="*60)

            # Check if subspace_cluster.py exists
            subspace_script = "subspace_cluster.py"
            if not os.path.exists(subspace_script):
                print(f"❌ Error: Subspace clustering script '{subspace_script}' not found")
                print("Please ensure subspace_cluster.py is in the current directory")
                return

            # Run subspace clustering
            try:
                import subprocess

                print(f"📂 Using imputed data file: {output_path}")

                # Intelligently determine dataset type based on output filename
                filename = os.path.basename(output_path).lower()
                dataset_name = None

                # Try to match COIL100
                if 'coil100' in filename:
                    if '_zeroed_20' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_20_subspace_gnn_imputed'
                    elif '_zeroed_30' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_30_subspace_gnn_imputed'
                    elif '_zeroed_40' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_40_subspace_gnn_imputed'
                    elif '_zeroed_50' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'coil100_zeroed_50_subspace_gnn_imputed'
                    elif '_zeroed' in filename and '_subspace_gnn_imputed' in filename: # Match without specific percentage
                        dataset_name = 'coil100_zeroed_subspace_gnn_imputed'
                    else: # Fallback for COIL100 if a specific pattern isn't matched
                        dataset_name = 'coil100_zeroed_gnn_imputed' 
                # Try to match ORL
                elif 'orl' in filename:
                    if '_zeroed_20' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_20_subspace_gnn_imputed'
                    elif '_zeroed_30' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_30_subspace_gnn_imputed'
                    elif '_zeroed_40' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_40_subspace_gnn_imputed'
                    elif '_zeroed_50' in filename and '_subspace_gnn_imputed' in filename:
                        dataset_name = 'orl_zeroed_50_subspace_gnn_imputed'
                    elif '_zeroed' in filename and '_subspace_gnn_imputed' in filename: # Match without specific percentage
                        dataset_name = 'orl_zeroed_subspace_gnn_imputed'
                    else: # Fallback for ORL if a specific pattern isn't matched
                        dataset_name = 'orl_zeroed_gnn_imputed'
                
                if dataset_name is None:
                    # Final default in case all above matches fail
                    dataset_name = 'coil100_random_zero_gnn_imputed'
                    print(f"⚠️ Could not infer exact dataset type from filename '{filename}', using default: {dataset_name}")

                print(f"🎯 Inferred dataset type: {dataset_name}")

                # Build command - main_subspace.py uses --data argument
                cmd = [sys.executable, subspace_script, "--data", dataset_name]

                print(f"🔧 Executing command: {' '.join(cmd)}")
                print("-" * 60)

                # Run subspace clustering script
                result = subprocess.run(cmd, capture_output=False, text=True)

                if result.returncode == 0:
                    print("-" * 60)
                    print("✅ Subspace clustering completed!")
                    print("🎯 Full pipeline: Data Imputation → Subspace Clustering successfully finished")
                else:
                    print("-" * 60)
                    print(f"❌ Subspace clustering failed with return code: {result.returncode}")
                    print("💡 Please check if subspace_cluster.py supports this dataset type")

            except Exception as e:
                print(f"❌ Error running subspace clustering: {e}")
                print("💡 You can manually run the following command:")
                # Here, provide manual command suggestion based on inferred dataset_name
                if dataset_name:
                    print(f"   python subspace_cluster.py --data {dataset_name}")
                else:
                    print(f"   Could not automatically generate manual run command. Please adjust dataset parameters manually based on the imputed file '{output_path}'.")
                
        else:
            print("\n💡 Hint: To automatically run subspace clustering, add the --auto_cluster parameter")
            print(f"   python run_subspace_gnn.py --input {args.input} --auto_cluster")
            print("   Full pipeline example:")
            print(f"   python run_subspace_gnn.py --input {args.input} --treat_zeros --auto_cluster")

        print("\n📊 You can also use the following command to view the imputation results:")
        print(f"  python view_mat_file_simple.py --file {output_path}")
        
    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
