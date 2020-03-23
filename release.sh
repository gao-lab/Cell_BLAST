#!/bin/bash

set -e

if [[ ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9a-z]+$ ]]; then
    echo "Usage: ./release_prep.sh x.y.z"
    exit 1
fi

echo "Current environment: $CONDA_PREFIX"
read -p "Confirm environment? [y/n] " yn
if [[ "$yn" != "y" ]]; then
    exit 2
fi

git pull
git status
read -p "Confirm repository is clean? [y/n] " yn
if [[ "$yn" != "y" ]]; then
    exit 3
fi

sed -i "s/version=.*/version=\"$1\",/g" setup.py
sed -i "s/__version__ = .*/__version__ = \"$1\"/g" Cell_BLAST/__init__.py

rm -rf docs/_build && sphinx-build -b html docs docs/_build
read -p "Confirm documentation is up-to-date? [y/n] " yn
if [[ "$yn" != "y" ]]; then
    exit 4
fi

(cd test && TEST_MODE=DEV ./test.sh)
read -p "Confirm all tests pass? [y/n] " yn
if [[ "$yn" != "y" ]]; then
    exit 5
fi

python setup.py sdist bdist_wheel
pip install dist/Cell_BLAST-$1.tar.gz
read -p "Confirm package can be installed? [y/n] " yn
if [[ "$yn" != "y" ]]; then
    exit 6
fi

(cd test && TEST_MODE=INSTALL ./test.sh)
read -p "Confirm all tests pass? [y/n] " yn
if [[ "$yn" != "y" ]]; then
    exit 7
fi

conda env export | awk -F= '
    BEGIN {
        mode = "chn";
    }
    /^(prefix|name):/ {
        next;
    }
    /^dependencies:$/ {
        mode = "dep";
        for (i = chn_idx - 1; i >= 0; ) {
            print chn_arr[i--];
        }
    }
    /(==|- pip:)/ {
        print "# "$0;
        next;
    }
    /^[^=]+=[^=]+=[^=]+$/ {
        print $1"="$2;
        next;
    }
    /- .+$/ {
        if (mode == "chn") {
            chn_arr[chn_idx++] = $0;
            next;
        }
    }
    {
        print $0;
    }
' > env.yml
git diff
read -p "Confirm changes? [y/n] " yn
if [[ "$yn" != "y" ]]; then
    exit 8
fi

git add .
git commit -m "Bump version to $1"
git push
git tag "v$1"
git push origin "v$1"
echo "Optionally, release this on Github and run 'twine upload dist/Cell_BLAST-$1*' to upload to PyPI."
exit 0
