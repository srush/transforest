#!/bin/bash

# [file=<feature_file>] id2feature "378218  379669:4"   [<feature_file>]

if [ "$file" = "" ]; then
	if [ "$2" != "" ]; then
		file=$2
	else
		file="$MODELS/features"
	fi
fi

echo "looking in "$file >> /dev/stderr
idstr=$1

ids=`echo $idstr | perl -ple 's/[:=][\-\d\.]+//g' | perl -ple 's/\s+/\\\\s\\|^/g'`
ids="^$ids\\s"

# becomes "378218\|379669" for grep

pcregrep $ids $file
