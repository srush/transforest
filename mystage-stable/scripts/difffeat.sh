#!/bin/bash

# difffeat.sh "378218:1        379669:1"    "378218:1        379624:1"

file=$MODELS/features
echo "looking in "$file >> /dev/stderr

strone=$1
strtwo=$2

ids=`echo $1 | perl -ple 's/[:=]\d+//g' | perl -ple 's/\s+/\\\|/g'`

grep $ids $file  > /tmp/feat-1

ids=`echo $2 | perl -ple 's/[:=]\d+//g' | perl -ple 's/\s+/\\\|/g'`

grep $ids $file  > /tmp/feat-2

diff -bd /tmp/feat-1 /tmp/feat-2
