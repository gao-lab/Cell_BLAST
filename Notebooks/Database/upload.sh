#!/bin/bash

set -e

if [ "${1}" = "test" ]; then
    echo "Uploading tarballs..."
    read -p "Confirm that safeweb:/rd1/home/caozj/aca has been properly backed up? [y/n] " yn
    if [[ "$yn" != "y" ]]; then exit 2; fi
    rsync -avzhPL --delete ./tarballs/* safeweb:/rd1/home/caozj/aca/
    echo "Tarballs uploaded! Please go to safeweb:/rd1/home/caozj/aca and run 'ls *.tar.gz | parallel -j4 -v tar xf {}'"
    echo "Updating MySQL database..."
    zcat aca.sql.gz | mysql -h 114.115.165.158 -u caozj -p aca_test
elif [ "${1}" = "release" ]; then
    echo "Uploading tarballs..."
    read -p "Confirm that cb:/rd1/aca has been properly backed up? [y/n] " yn
    if [[ "$yn" != "y" ]]; then exit 2; fi
    rsync -avzhPL --delete ./tarballs/* cb:/rd1/aca/
    echo "Tarballs uploaded! Please go to cb:/rd1/aca and run 'ls *.tar.gz | parallel -j4 -v tar xf {}'"
    echo "Updating MySQL database..."
    zcat aca.sql.gz | mysql -h 114.115.165.158 -u caozj -p aca
else
    echo "Usage: ./upload.sh test|release"
    exit 1
fi

exit 0
