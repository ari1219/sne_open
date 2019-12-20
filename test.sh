#!/bin/sh

echo "start training"
python src/train_each_epoch.py slashdot1 --epoch 3
echo "finish training"
echo "caliculate accuracy of relation prediction"
python src/relation_prediction.py slashdot1
echo "caliculate accuracy of link prediction"
python src/link_prediction.py slashdot1
