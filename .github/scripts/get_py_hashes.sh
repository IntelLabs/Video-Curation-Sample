# pip-compile is part of pip-tools pypi package

# FRONTEND
pip-compile -o frontend/requirements.txt --generate-hashes frontend/requirements.in

# UDF
pip-compile -o udf/requirements.txt --generate-hashes udf/requirements.in

# VIDEO
pip-compile -o video/requirements.txt --generate-hashes video/requirements.in

# pip-compile -o video/requirements.CPU.txt --generate-hashes video/requirements.CPU.in
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip compile video/requirements.CPU.in -o video/requirements.CPU.txt --generate-hashes

pip-compile -o video/requirements.GPU.txt --generate-hashes video/requirements.GPU.in
