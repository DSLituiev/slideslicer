#!/bin/bash

# LINK images for binary classification (inflammation / normal)
#INDIR=~/data_1024/fullsplit
BASE=$1
TAG=glom
#BASE=../data/data_1024
INDIR=$BASE/fullsplit/all
OUTDIR=$BASE/glom_split/all
echo -e "linking from\t$INDIR"
echo -e "linking to\t$OUTDIR"
mkdir -p $INDIR
# mv ~/data_1024/* $INDIR
mkdir -p $OUTDIR
mkdir -p $OUTDIR/glom
mkdir -p $OUTDIR/normal

# link inflammation patches
#ln -s $INDIR/*infl*/*png $OUTDIR/infl

find $INDIR/*$TAG*/*png  -exec sh -c 'ln -s $(readlink -f $1) '$OUTDIR/$TAG'' _ {} \;
find $INDIR/*$TAG*/*json  -exec sh -c 'ln -s $(readlink -f $1) '$OUTDIR/$TAG'' _ {} \;


# link everything else
dirs=$(ls $INDIR)
for dd in ${dirs[@]}
do
    if [[ $dd == *"$TAG"* ]]; then
#         echo "$dd"
        continue
    else
        echo "${dd}"
        #ln -s $INDIR/${dd}/*png $OUTDIR/normal/
        find $INDIR/${dd}/*png -exec sh -c 'ln -s $(readlink -f $1) '$OUTDIR/normal/'' _ {} \;
        find $INDIR/${dd}/*json -exec sh -c 'ln -s $(readlink -f $1) '$OUTDIR/normal/'' _ {} \;
    fi
done

