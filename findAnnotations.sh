DIR=$1
KEYWORD='glom'

echo -e "scanning directory\t$DIR" >&2

find $DIR -type f  -name "*.xml" -print0 | \
    xargs -r0 -P2 grep -nHil -m1 $KEYWORD | \
    sed "s|$DIR||"

