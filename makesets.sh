CLASSES=(infl normal)
#DATADIR=../data/data_1024/infl_split/
#DATADIR=/repos/data/data_128_subsample_8x/fullsplit/

#CLASSES=(glom normal)

#DATADIR=../data/data_1024/glom_split/
DATADIR="$1"
cd $DATADIR;
DATADIR=$(pwd -P)
cd -

echo "CLASSES:"
for CLASS in ${CLASSES[@]}
do
    echo -e "\t$CLASS"
done

test_ids=(70bb3032750d09e7549928c0dbf79afc30d7cb68 a1fc67fbb21f43b9e8904b9b46bd94f83493b37a f7f931a5cf3185a385e9aa34e6e9a566fc88000)
trainids=(c886827fe8c10b4699a0f1616331e36b46a05617 dfe3ee768f72bddd289c7d5bb88b15cbb89be7e6)

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

for CLASS in ${CLASSES[@]}
do
    mkdir -p ${DATADIR}/val/${CLASS}
    mkdir -p ${DATADIR}/train/${CLASS}

    for id in ${trainids[@]}
    do
        #echo ""
        #ln -s ${DATADIR}/all/${CLASS}/${id}* ${DATADIR}/train/${CLASS}/
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'echo $PWD/$(readlink $1) ' _ {} \;
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $PWD/$(readlink $1) '${DATADIR}/train/${CLASS}/'' _ {} \;
		find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $(readlink -f $1) '${DATADIR}/train/${CLASS}/'' _ {} \;
    done
    echo -e "see\t${DATADIR}/train/"

    for id in ${test_ids[@]}
    do
        #ln -s ${DATADIR}/all/${CLASS}/${id}* ${DATADIR}/val/${CLASS}/
		find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $(readlink -f $1) '${DATADIR}/test/${CLASS}/'' _ {} \;
        #find ${DATADIR}/all/${CLASS}/ -name "${id}*" -exec sh -c 'ln -s $PWD/$(readlink $1) '${DATADIR}/val/${CLASS}/'' _ {} \;
    done
    echo -e "see\t${DATADIR}/val/"
done
