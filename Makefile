
build:
	conda create --file env.yml --prefix /home/graphicslab/miniconda3/envs/tf1.13gpu

ANNFASS_SPLIT:=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/annfass_splits_semifinal
BUILDNET_SPLIT:=/media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/logs/buildnet_reconstruction_splits
test-buildnet:
	export LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64/:/usr/local/cuda-10.0/include:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.1/include:/usr/local/cuda-10.1/lib64: \
		&& /home/graphicslab/.conda/envs/pointnetae/bin/python test.py \
		--data_path $(BUILDNET_SPLIT)/ply_10K/split_train_val_test_debug \
		--eval_dir evaluation
test-annfass:
	export LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64/:/usr/local/cuda-10.0/include:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.1/include:/usr/local/cuda-10.1/lib64: \
		&& /home/graphicslab/.conda/envs/pointnetae/bin/python test.py \
			--data_path $(ANNFASS_SPLIT)/ply_10K/split_train_val_test \
			--eval_dir annfass_evaluation

