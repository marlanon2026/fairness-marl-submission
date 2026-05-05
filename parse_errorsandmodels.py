import re
from pathlib import Path
import csv

# Path to your logs directory
logs_dir = Path("./logs_june")

# List of error patterns to look for
error_patterns = [
    r"OSError: \[Errno 5\] Input/output error",
    r"No such file or directory",
    r"Missing PDDL file",
    r"No space left on device",
    r"Disk space error",
    r"NoneType.*has no attribute.*groups",
    r"PDDL parsing error",
    r"Unable to locate a modulefile for 'cuda",
    r"CUDA module error",
    r"Named config not found",
    r"Sacred configuration error",
]

error_regexes = [re.compile(p, re.IGNORECASE) for p in error_patterns]

# Regex to extract model directory from line
model_dir_pattern = re.compile(r"my_main Saving models to (results/models/\S+)")

# Store results as a list of tuples (job_id, model_dir)
results = []

# Loop over all .err files
for err_file in logs_dir.glob("*.err"):
    try:
        content = err_file.read_text()
    except Exception as e:
        print(f"Could not read {err_file}: {e}")
        continue

    # Check if any error matches
    if any(regex.search(content) for regex in error_regexes):
        # Extract job ID
        m = re.search(r"_(\d+)\.err$", err_file.name)
        if not m:
            print(f"Could not extract job ID from filename: {err_file.name}")
            continue
        job_id = m.group(1)

        # Extract model directory (first match)
        model_match = model_dir_pattern.search(content)
        model_dir = model_match.group(1) if model_match else ""

        # Append to results
        results.append((job_id, model_dir))

# Write to CSV
csv_path = logs_dir / "error_runs_with_models.csv"
with csv_path.open("w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["job_id", "model_dir"])
    for job_id, model_dir in results:
        writer.writerow([job_id, model_dir])

print(f"\n✅ Done. Saved CSV: {csv_path}")
