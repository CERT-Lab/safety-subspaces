instruct_path="give_instruct_path_here"
useful_path="give_useful_path_here"
harmful_path="give_harmful_path_here"
granularity="give_granularity_here"
output_csv="give_output_csv_here"

# Run commands with error handling
run_command() {
  echo "Running: $1"
  if ! eval "$1"; then
    echo "Command failed: $1"
    return 1
  fi
  echo "Command completed successfully"
  return 0
}

run_command "CUDA_VISIBLE_DEVICES=0 python update_spaces/update_spaces_functions.py --instruct-path \"$instruct_path\" --useful-path \"$useful_path\" --harmful-path \"$harmful_path\" --granularity tensor --output-csv \"$output_csv\""








