#!/bin/bash -e
#######################################################################################################################
# THIS SCRIPT DISPLAYS DETECTIONS OF INTEREST
#
# INPUTS:
#   - list_of_objects (required): comma-delimited list of objects [Default: person]
#######################################################################################################################
list_of_objects="${1:-person}"

echo "list_of_objects: ${list_of_objects}"

# SPLIT LIST INTO ARRAY
IFS="," read -ra parts <<< "$list_of_objects"

# PREPEND " detected" TO EACH OBJECT
new_list=()
for obj in "${parts[@]}"; do
    new_list+=("${obj} detected")
done

# CREATE REGEX TO RETURN RESULTS
IFS="|"
grep_substrings="${new_list[*]}"
echo "grep_substrings: $grep_substrings"

# RETRIEVE OBJS FROM LOGS
# -f: Follow log
# -t: Include timestamp
docker logs -f --tail 10 lcc_video-service_1 | grep -E "$grep_substrings"

