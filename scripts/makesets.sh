#!/bin/bash

#CLASSES=(infl normal)
#DATADIR=../data/data_1024/infl_split/
#DATADIR=/repos/data/data_128_subsample_8x/fullsplit/

#CLASSES=(glom normal)

#DATADIR=../data/data_1024/glom_split/
DATADIR="$1"
cd $DATADIR;
DATADIR=$(pwd -P)
cd -

CLASSES=(lymph norm)
echo "CLASSES:"
for CLASS in ${CLASSES[@]}
do
    echo -e "\t$CLASS"
done

#val_ids=(70bb3032750d09e7549928c0dbf79afc30d7cb68 a1fc67fbb21f43b9e8904b9b46bd94f83493b37a f7f931a5cf3185a385e9aa34e6e9a566fc88000)
#trainids=(c886827fe8c10b4699a0f1616331e36b46a05617 dfe3ee768f72bddd289c7d5bb88b15cbb89be7e6)

SETFILE=~/repos/tables/splits-2019-03-24.csv

readarray train_ids < <(grep 'train' $SETFILE | cut -f1 -d',' )
readarray test_ids < <(grep 'test' $SETFILE | cut -f1 -d',' )
readarray val_ids < <(grep 'val' $SETFILE | cut -f1 -d',' )

echo -e "test\t${test_ids[1]}"
#val_ids=(01c02ba0782776e0aee49c751d170fb5a7f3e189 1613eab8267591c8aad6db6c18308c970bdb0229 4aeba5a0fd1fd39eb494ee759c8d3d566a30bf63 56356da1a0fa39a7394f83e45ff50bd0f54d9461 8250cc470d1320502cfdd7022d410de161885945 88f73323af6f45440a7a532dc4f5aca51c624f5c)

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

for CLASS in ${CLASSES[@]}
do
    mkdir -p ${DATADIR}/val/${CLASS}
    mkdir -p ${DATADIR}/train/${CLASS}
    mkdir -p ${DATADIR}/test/${CLASS}

    for id in ${test_ids[@]}
    do
        echo ""
        echo -e "ID:\t$id"
        #ln -s ${DATADIR}/all/${CLASS}/${id}* ${DATADIR}/train/${CLASS}/
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'echo $PWD/$(readlink $1) ' _ {} \;
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $PWD/$(readlink $1) '${DATADIR}/train/${CLASS}/'' _ {} \;
		find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $(readlink -f $1) '${DATADIR}/test/${CLASS}/'' _ {} \;
    done
    echo -e "see\t${DATADIR}/test/"

    for id in ${train_ids[@]}
    do
        #echo ""
        #ln -s ${DATADIR}/all/${CLASS}/${id}* ${DATADIR}/train/${CLASS}/
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'echo $PWD/$(readlink $1) ' _ {} \;
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $PWD/$(readlink $1) '${DATADIR}/train/${CLASS}/'' _ {} \;
		find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $(readlink -f $1) '${DATADIR}/train/${CLASS}/'' _ {} \;
    done
    echo -e "see\t${DATADIR}/train/"

    for id in ${val_ids[@]}
    do
        #ln -s ${DATADIR}/all/${CLASS}/${id}* ${DATADIR}/val/${CLASS}/
		find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $(readlink -f $1) '${DATADIR}/val/${CLASS}/'' _ {} \;
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $PWD/$(readlink $1) '${DATADIR}/val/${CLASS}/'' _ {} \;
    done
    echo -e "see\t${DATADIR}/val/"
done
