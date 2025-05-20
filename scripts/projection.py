import subprocess
from pathlib import Path


def main() -> None:
    # Adjust this if your original script lives elsewhere or has a different name
    target_script = "./scripts/projection_helper.py"

    path_to_random_script = "./utils/post_processing_subspace_random.py"
    path_to_random_k_script = "./utils/post_processing_subspace_random_k.py"
    path_to_top_k_script = "./utils/post_processing_subspace.py"

    model_name = "model_name" #Change this

    finetuned_model_dict = {"Full_Useful": "path_to_full_useful_model", 
                            "Full_Harmful": "path_to_full_harmful_model", 
                            "Contaminated": "path_to_contaminated_model", 

    }  #Change this

    merged_script_dict = { 
        "Random": path_to_random_script,
        "RandomK": path_to_random_k_script,
        "TopK": path_to_top_k_script
    }
    run_base_models = 0
    for finetuned_model in  finetuned_model_dict.keys(): 
        for merged_script in merged_script_dict.keys():
            print(f"Starging the script with finetuned_model: {finetuned_model} and merged_script: {merged_script}") 
            results_csv = f"path_to_results_csv/{model_name}/results_{finetuned_model}_{merged_script}_{model_name}.csv"

            cmd = [
                "python",
                str(target_script),
                "--base_model_path", "meta-llama/Llama-3.2-1B", #Change this
                "--aligned_model_path", "meta-llama/Llama-3.2-1B-Instruct", #Change this
                "--finetuned_model_path", finetuned_model_dict[finetuned_model],
                "--method_types", "same,orth",
                "--fracs", "0.01,0.25,0.5,0.75,0.99",
                "--output_csv", results_csv,
                "--merge_script_path", merged_script_dict[merged_script],
                "--run_base_models", str(run_base_models)
            ]

    # Run the command and forward stdout/stderr to your terminal
            subprocess.run(cmd, check=True)
            run_base_models = 0


if __name__ == "__main__":
    main()
