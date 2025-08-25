#!/bin/bash -e

# UDF server
cd /home/remote_function/
python3 udf_server.py ${UDF_PORT} .
# sleep 10
