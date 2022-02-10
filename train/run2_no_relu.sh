# ReFine
python3 refine_train_no_relu.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4
python3 refine_train_no_relu.py --dataset mnist --hid 50 --epoch 25 --ratio 0.2 --lr 1e-3
python3 refine_train_no_relu.py --dataset mutag --hid 100 --epoch 25 --ratio 0.4 --lr 1e-3

python3 pg_train_no_relu.py --dataset ba3 --hid 50 --epoch 25 --ratio 0.4 --lr 1e-4
python3 pg_train_no_relu.py --dataset mnist --hid 50 --epoch 25 --ratio 0.2 --lr 1e-3
python3 pg_train_no_relu.py --dataset mutag --hid 100 --epoch 25 --ratio 0.4 --lr 1e-3
