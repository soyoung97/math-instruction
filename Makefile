normal:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --mode normal --batch-size 32 --log_dir logs/normal

explain:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --mode explain --batch-size 32 --log_dir logs/explain
