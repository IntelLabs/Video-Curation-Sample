# pip-compile is part of pip-tools pypi package

# FASTAPI
pip-compile -o fastapi/requirements.txt --generate-hashes fastapi/requirements.in

# pip-compile -o fastapi/requirements.CPU.txt --generate-hashes fastapi/requirements.CPU.in
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip compile fastapi/requirements.CPU.in -o fastapi/requirements.CPU.txt --generate-hashes

pip-compile -o fastapi/requirements.GPU.txt --generate-hashes fastapi/requirements.GPU.in

# FINETUNE
pip-compile -o finetune/requirements.txt --generate-hashes finetune/requirements.in

# FRONTEND
pip-compile -o frontend/requirements.txt --generate-hashes frontend/requirements.in

# UDF
pip-compile -o udf/requirements.txt --generate-hashes udf/requirements.in

# VIDEO
pip-compile -o video/requirements.txt --generate-hashes video/requirements.in