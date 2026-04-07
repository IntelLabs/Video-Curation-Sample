# pip-compile is part of pip-tools pypi package

# FRONTEND
pip-compile -o frontend/requirements.txt --generate-hashes frontend/requirements.in

# UDF
pip-compile -o udf/requirements.txt --generate-hashes udf/requirements.in

# VIDEO
# curl -LsSf https://astral.sh/uv/install.sh | sh

# pip-compile -o video/requirements.txt --generate-hashes video/requirements.in
uv pip compile video/requirements.in -o video/requirements.txt --generate-hashes --index-strategy unsafe-best-match

# pip-compile -o video/requirements.CPU.txt --generate-hashes video/requirements.CPU.in
uv pip compile video/requirements.CPU.in -o video/requirements.CPU.txt --generate-hashes --index-strategy unsafe-best-match

pip-compile -o video/requirements.GPU.txt --generate-hashes video/requirements.GPU.in
