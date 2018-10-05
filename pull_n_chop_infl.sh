#!/bin/bash

#cell#


# COPY the data to local disk
# sudo chown -R dlituiev /home/dlituiev/.gsutil/tracker-files
TMPDIR=/tmp/kidney
XMLFILELIST=./xmlfiles.list
mkdir -p $TMPDIR 
LINE=2
for LINE in `seq 1 5`
do
    # PREV_SLIDEID=$(head  /home/dlituiev/metadata/xmlfiles.list | sed -n -e $((${LINE}-1))p | sed 's/\.xml//' )
    # rm "$HOME/tmpdata/${PREV_SLIDEID}.svs"
    SLIDEID=$(head  $XMLFILELIST | sed -n -e ${LINE}p | sed 's/\.xml//' ) || exit 1
    echo $SLIDEID

    SVS="$TMPDIR/${SLIDEID}.svs"
    XML="$TMPDIR/${SLIDEID}.xml"

    if [ -a "$SVS" ] ; then
        echo -e "${SVS}\talready exists"
    else
        gsutil cp gs://kidney-rejection/${SLIDEID}.xml $TMPDIR 
        gsutil cp gs://kidney-rejection/${SLIDEID}.svs $TMPDIR
    fi

    python3 $PWD/sample_from_slide.py --data-root "../data/" --all-grid $XML && rm $SVS
done

if [ $RMTMP ]
then
    rm -rf $TMPDIR
fi
