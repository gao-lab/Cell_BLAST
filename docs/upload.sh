#!/bin/bash

set -e

if [ "${1}" = "test" ]; then
    target="safeweb:/rd1/home/cb/public/doc/"
elif [ "${1}" = "release" ]; then
    target="cb:/home/cb/public/doc/"
else
    echo "Usage: ./upload.sh test|release"
    exit 1
fi

read -p "Confirm that ${target} has been properly backed up? [y/n] " yn
if [[ "$yn" != "y" ]]; then exit 2; fi
rsync -avzhPL --delete ./_build/* ${target}
echo "Finished!"
 
exit 0
