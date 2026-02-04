import os
from huggingface_hub import snapshot_download

# CONFIGURATION
DESTINATION_DIR = "MatrixCity"  # Where the files will go
REPO_ID = "BoDai/MatrixCity"

print(f"Starting download to: {os.path.abspath(DESTINATION_DIR)}")

# ---------------------------------------------------------
# STEP 1: Download Coordinates (JSONs)
# ---------------------------------------------------------
print("--> Downloading Metadata (Coordinates)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="small_city/street/*.json",
    local_dir=DESTINATION_DIR,
    resume_download=True
)

# ---------------------------------------------------------
# STEP 2: Download Test Set (Queries)
# ---------------------------------------------------------
print("--> Downloading Test Set (Queries)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="small_city/street/test/*",
    local_dir=DESTINATION_DIR,
    resume_download=True
)

# ---------------------------------------------------------
# STEP 3: Download Training Set (Database)
# ---------------------------------------------------------

# OPTION A: Download ONLY Block 1 (Safe start ~2GB)
#print("--> Downloading Train Set (Block 1 Only)...")
#snapshot_download(
#    repo_id=REPO_ID,
#    repo_type="dataset",
#    allow_patterns="small_city/street/train/block_1.tar",
#    local_dir=DESTINATION_DIR,
#    resume_download=True
#)

# OPTION B: Download EVERYTHING (Uncomment to get full ~400GB)
print("--> Downloading ALL Train Blocks (Warning: Huge)...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns="small_city/street/train/*",
    local_dir=DESTINATION_DIR,
    resume_download=True
)

print("Small City download complete!")
