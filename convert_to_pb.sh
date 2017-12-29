[ -a ./build_image_data.py ] || \
    curl -O https://raw.githubusercontent.com/tensorflow/models/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/inception/inception/data/build_image_data.py

DATADIR=../data/data_1024/infl_split

echo "normal" > $DATADIR/labels.txt
echo "infl" >> $DATADIR/labels.txt

rm -rf $DATADIR/protobuf
mkdir -p $DATADIR/protobuf

python3 build_image_data.py \
    --train_directory  $DATADIR/train \
    --validation_directory $DATADIR/val \
    --output_directory $DATADIR/protobuf \
    --train_shards 1 \
    --validation_shards 1 \
    --num_threads 1 \
    --labels_file $DATADIR/labels.txt
