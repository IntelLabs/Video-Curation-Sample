# pip-compile is part of pip-tools pypi package
# UV is required for hashes only and can be installed via `curl -LsSf https://astral.sh/uv/install.sh | sh`

# Get absolute path of the script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
GH_DIR=$(dirname "${SCRIPT_DIR}")
REPO_DIR=$(dirname "${GH_DIR}")

# FASTAPI
uv pip compile ${GH_DIR}/assets/fastapi/requirements.CPU.in --no-header --no-annotate -o ${REPO_DIR}/fastapi/requirements.CPU.txt --generate-hashes --allow-unsafe --index-strategy unsafe-best-match
pip-compile --no-header --no-annotate -o ${REPO_DIR}/fastapi/requirements.GPU.txt --generate-hashes --allow-unsafe ${GH_DIR}/assets/fastapi/requirements.GPU.in

# FINETUNE
pip-compile --no-header --no-annotate -o ${REPO_DIR}/finetune/requirements.txt --generate-hashes --allow-unsafe ${GH_DIR}/assets/finetune/requirements.in

# FRONTEND
pip-compile --no-header --no-annotate -o ${REPO_DIR}/frontend/requirements.txt --generate-hashes --allow-unsafe ${GH_DIR}/assets/frontend/requirements.in

# UDF
pip-compile --no-header --no-annotate -o ${REPO_DIR}/udf/requirements.txt --generate-hashes --allow-unsafe ${GH_DIR}/assets/udf/requirements.in

# VIDEO
pip-compile --no-header --no-annotate -o ${REPO_DIR}/video/requirements.txt --generate-hashes --allow-unsafe ${GH_DIR}/assets/video/requirements.in
