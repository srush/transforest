#!/bin/bash

# getsentbyid #894

id=`expr $1 \* 2 + 3`

echo $id >/dev/stderr

head -$id sec23.1best | tail -1 
