gen2:
	CUDA_VISIBLE_DEVICES=2 python3 generate.py --mode explain --path logs/explain-prev/last.ckpt --output outputs/explain-prev-last.txt 
gen1:
	CUDA_VISIBLE_DEVICES=1 python3 generate.py --mode explain --path logs/explain-prev/last.ckpt --output outputs/explain-prev-last.txt
gen0:
	CUDA_VISIBLE_DEVICES=0 python3 generate.py --mode explain --path logs/explain-prev/last.ckpt --output outputs/explain-prev-last.txt
normal:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --mode normal --batch-size 32 --log_dir logs/normal

explain:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py --mode explain --batch-size 32 --log_dir logs/explain
