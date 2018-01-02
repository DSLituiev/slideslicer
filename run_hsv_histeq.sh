CLASSES=(infl normal)
DATADIR=../data/data_1024/fullsplit/
cd $DATADIR;
DATADIR=$(pwd -P)
cd -

find ${DATADIR} -name "*.png" -exec \
    python3 hsv_histeq.py {} \;

