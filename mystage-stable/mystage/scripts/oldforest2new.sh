perl -ple 's/^\t(\S+)\s+(.*)/\t\2 ||| 0=\1/g; s/^(\d+)\t(.*)\t(\S+)\s(.*)/\1\t\2\t\4 ||| \5/g'
