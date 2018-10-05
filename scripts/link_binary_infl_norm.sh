#!/bin/bash

# LINK images for binary classification (inflammation / normal)
#INDIR=~/data_1024/fullsplit
BASE=$1
#BASE=../data/data_1024
INDIR=$BASE/fullsplit/all
OUTDIR=$BASE/infl_split/all
echo -e "linking from\t$INDIR"
echo -e "linking to\t$OUTDIR"
# mv ~/data_1024/* $INDIR
mkdir -p $OUTDIR
mkdir -p $OUTDIR/infl
mkdir -p $OUTDIR/normal

# link inflammation patches
#ln -s $INDIR/*infl*/*png $OUTDIR/infl

find $INDIR/*infl*/*png  -exec sh -c 'ln -s $(readlink -f $1) '$OUTDIR/infl'' _ {} \;


# link everything else
dirs=$(ls $INDIR)
for dd in ${dirs[@]}
do
    if [[ $dd == *"infl"* ]]; then
#         echo "$dd"
        continue
    else
        echo "${dd}"
        #ln -s $INDIR/${dd}/*png $OUTDIR/normal/
        find $INDIR/${dd}/*png -exec sh -c 'ln -s $(readlink -f $1) '$OUTDIR/normal/'' _ {} \;
    fi
done

