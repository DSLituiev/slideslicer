#!/bin/bash

#cell#


# COPY the data to local disk
# sudo chown -R dlituiev /home/dlituiev/.gsutil/tracker-files
TMPDIR=~/tmpdata/
mkdir -p $TMPDIR 
LINE=2
for LINE in `seq 2 2`
do
    # PREV_SLIDEID=$(head  /home/dlituiev/metadata/xmlfiles.list | sed -n -e $((${LINE}-1))p | sed 's/\.xml//' )
    # rm "$HOME/tmpdata/${PREV_SLIDEID}.svs"
    SLIDEID=$(head  /home/dlituiev/metadata/xmlfiles.list | sed -n -e ${LINE}p | sed 's/\.xml//' )
    echo $SLIDEID

    SVS="$TMPDIR/${SLIDEID}.svs"
    XML="$TMPDIR/${SLIDEID}.xml"

    if [ -a "$SVS" ] ; then
        echo -e "${SVS}\talready exists"
    else
        gsutil cp gs://kidney-rejection/${SLIDEID}.xml $TMPDIR 
        gsutil cp gs://kidney-rejection/${SLIDEID}.svs $TMPDIR
    fi

    python3 $HOME/slideslicer/sample_from_slide.py $XML && rm $SVS
done


# LINK images for binary classification (inflammation / normal)

INDIR=~/data_1024/fullsplit
OUTDIR=~/data_1024/infl_split/all
mkdir -p $INDIR
# mv ~/data_1024/* $INDIR
mkdir -p $OUTDIR
mkdir -p $OUTDIR/infl
mkdir -p $OUTDIR/normal
ln -s $INDIR/*infl*/*png $OUTDIR/infl


dirs=$(ls $INDIR)
for dd in ${dirs[@]}
do
    if [[ $dd == *"infl"* ]]; then
#         echo "$dd"
        continue
    else
        echo "${dd}"
        ln -s $INDIR/${dd}/*png $OUTDIR/normal/
    fi
done

