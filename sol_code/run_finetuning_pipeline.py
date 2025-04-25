import os
import argparse
import logging
import subprocess
import time
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"finetuning_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def run_command(cmd, desc=None, wait=True):
    """Run a shell command and log output."""
    if desc:
        logging.info(f"Running: {desc}")
    logging.info(f"Command: {cmd}")
    
    try:
        if wait:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Stream output
            for stdout_line in iter(process.stdout.readline, ""):
                if stdout_line:
                    logging.info(stdout_line.strip())
            
            for stderr_line in iter(process.stderr.readline, ""):
                if stderr_line:
                    logging.error(stderr_line.strip())
            
            process.stdout.close()
            process.stderr.close()
            return_code = process.wait()
            
            if return_code != 0:
                logging.error(f"Command failed with return code {return_code}")
                return False
            
            return True
        else:
            # For SBATCH jobs, we don't wait
            subprocess.run(cmd, shell=True, check=True)
            return True
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False

def setup_environment():
    """Install required packages."""
    logging.info("Setting up environment...")
    
    packages = [
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "scipy>=1.11.0",
        "sentencepiece>=0.1.99",
        "tensorboard>=2.14.0",
        "tqdm>=4.66.0"
    ]
    
    # Install each package
    for package in packages:
        cmd = f"pip install {package}"
        run_command(cmd, f"Installing {package}")
    
    return True

def setup_datasets():
    """Set up and process datasets."""
    logging.info("Setting up datasets...")
    return run_command("python data/setup_datasets.py", "Setting up datasets")

def run_trainings():
    """Submit training jobs to SLURM."""
    logging.info("Submitting training jobs...")
    
    # Submit general fine-tuning
    general_cmd = "sbatch training/sbatch/train_enhanced_general.sh"
    general_success = run_command(general_cmd, "Submitting general fine-tuning job", wait=False)
    
    # Wait a moment to ensure job is submitted
    time.sleep(5)
    
    # Get the job ID from squeue
    get_job_id_cmd = "squeue -u $USER -o \"%j %i\" | grep enhanced_general | awk '{print $2}'"
    process = subprocess.Popen(get_job_id_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    general_job_id = stdout.decode().strip() if stdout else None
    logging.info(f"General fine-tuning job ID: {general_job_id}")
    
    # Submit math fine-tuning with dependency on general job
    math_cmd = f"sbatch"
    if general_job_id:
        math_cmd += f" --dependency=afterany:{general_job_id}"
    math_cmd += " training/sbatch/train_enhanced_math.sh"
    math_success = run_command(math_cmd, "Submitting math fine-tuning job", wait=False)
    
    time.sleep(5)
    
    # Get math job ID
    get_math_id_cmd = "squeue -u $USER -o \"%j %i\" | grep enhanced_math | awk '{print $2}'"
    process = subprocess.Popen(get_math_id_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    math_job_id = stdout.decode().strip() if stdout else None
    logging.info(f"Math fine-tuning job ID: {math_job_id}")
    
    # Submit reasoning fine-tuning with dependency on math job
    reasoning_cmd = f"sbatch"
    if math_job_id:
        reasoning_cmd += f" --dependency=afterany:{math_job_id}"
    reasoning_cmd += " training/sbatch/train_enhanced_reasoning.sh"
    reasoning_success = run_command(reasoning_cmd, "Submitting reasoning fine-tuning job", wait=False)
    
    return general_success and math_success and reasoning_success

def run_evaluation():
    """Run evaluation on the fine-tuned models."""
    logging.info("Running evaluation...")
    return run_command(
        "python evaluation/eval.py --evaluate_all --data_path data/raw/dev_data.json --output_dir evaluation/results",
        "Running evaluation on all models"
    )

def main():
    parser = argparse.ArgumentParser(description="Run enhanced fine-tuning pipeline")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training jobs")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    # Setup environment
    if not args.skip_setup:
        setup_success = setup_environment()
        if not setup_success:
            logging.error("Environment setup failed")
            return
    
    # Setup datasets
    if not args.skip_datasets:
        dataset_success = setup_datasets()
        if not dataset_success:
            logging.error("Dataset setup failed")
            return
    
    # Run training
    if not args.skip_training:
        training_success = run_trainings()
        if not training_success:
            logging.error("Training submission failed")
            return
    
    # Run evaluation
    if not args.skip_eval:
        eval_success = run_evaluation()
        if not eval_success:
            logging.error("Evaluation failed")
            return
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f"Pipeline completed successfully!")
    logging.info(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    main()