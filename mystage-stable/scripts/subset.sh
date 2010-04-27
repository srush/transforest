nametag=$1

grep "`echo -e $1`" features > f-$2
num=`wc -l f-$2 | cut -f 1 -d " "`
lastid=`tail -1 f-$2 | cut -f 1`
lastid=`expr $lastid + 1`
cat weights | head -$lastid | tail -$num > w-$2

# johnson's code requires gzipped input
gzip -c f-$2 > f-$2.gz
gzip -c w-$2 > w-$2.gz
