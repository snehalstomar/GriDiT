#=========Setup=========================
setup:
	conda env create -f gridit_env.yaml
#=======================================

#========================================Stage-1-Inference========================================
NUM_NODES ?= 1
NUM_GPUS ?= 1
AVAILABLE_GPUS ?= 3
IMG_SIZE ?= 512
CKPT_PATH_INFER_STAGE_1 ?= ckpts/stage1_sky_ckpt.pt # path_to_stage 1 ckpt
NUM_SEQUENCES ?= 2 #Required number of synthetic sequences  
RANDOM_SEED ?= 45
NUM_SAMPLING_STEPS ?= 250
GRID_SZ_K ?= 4
SAMPLING_FRAMES_LEN ?= 64 #Required length of each synthetic sequences 
VID_FPS ?= 4

sample_long_sequences:
	CUDA_VISIBLE_DEVICES=$(AVAILABLE_GPUS) python -m scripts.sample_n_save_different --model DiT-XL/2 --image-size $(IMG_SIZE) --ckpt $(CKPT_PATH_INFER_STAGE_1) --target_dir outputs/starting_grid_samples --nsamples $(NUM_SEQUENCES) --seed $(RANDOM_SEED) --num-sampling-steps $(NUM_SAMPLING_STEPS)
	CUDA_VISIBLE_DEVICES=$(AVAILABLE_GPUS) python -m scripts.sample_n_step_from_folder_three_row_4x4 --model DiT-XL/2 --image-size $(IMG_SIZE) --ckpt $(CKPT_PATH_INFER_STAGE_1)  --input-dir outputs/starting_grid_samples --target-dir outputs/stage1_longer_sampling --vidLength $(SAMPLING_FRAMES_LEN) --gridSz $(GRID_SZ_K) --num-sampling-steps $(NUM_SAMPLING_STEPS)
	CUDA_VISIBLE_DEVICES=$(AVAILABLE_GPUS) python -m scripts.sample_n_step_from_folder_three_row_4x4_stage2 --model DiT-XL/2 --image-size $(IMG_SIZE) --ckpt $(CKPT_PATH_INFER_STAGE_1) --input-dir outputs/stage1_longer_sampling --target-dir outputs/stage2_longer_sampling --vidLength $(SAMPLING_FRAMES_LEN) --gridSz $(GRID_SZ_K) --num-sampling-steps $(NUM_SAMPLING_STEPS)
	python -m src.utils.grid_splitter_vid_maker_stage2_att3 --inputDir outputs/stage2_longer_sampling --targetDir outputs/splitted_output --fps $(VID_FPS) --condType three
#===================================================================================================

#===========Training============================================================
MASTER_PORT ?= 29500
MASTER_PORT_SR ?= 29503
IMG_SIZE ?= 512
BATCH_SIZE ?= 2
CKPT_EVERY_N_ITERS ?= 10
LOG_EVERY_N_ITERS ?= 10
CKPT_PATH_1 ?= ckpts/sktimelipse1
CKPT_PATH_2 ?= ckpts/sktimelipse2
NUM_GRAD_ACCUMULATION_ITERS ?= 16
TRAINING_DSET_PATH_STAGE1 ?= #training_dataset_path_stage_1
TRAINING_DSET_PATH_STAGE2 ?= #training_dataset_path_stage_2
PRETRAINED_CKPT_PATH ?= ckpts/pretrained/ #path to pretrained vanilla DiT models.
SR_SCLAE ?= 4

train-stage-1:
	CUDA_VISIBLE_DEVICES=$(AVAILABLE_GPUS) torchrun --nnodes=$(NUM_NODES) --nproc_per_node=$(NUM_GPUS) --master_port=$(MASTER_PORT) -m scripts.train_warmup_bs_accumulation --model DiT-XL/2 --image-size $(IMG_SIZE) --num-classes 1 --global-batch-size $(BATCH_SIZE) --ckpt-every $(CKPT_EVERY_N_ITERS) --log-every $(LOG_EVERY_N_ITERS) --results-dir $(CKPT_PATH_1) --data-path $(TRAINING_DSET_PATH_STAGE1) --accumulation-steps $(NUM_GRAD_ACCUMULATION_ITERS) --pretrained_ckpt_path $(PRETRAINED_CKPT_PATH)

train-stage-2:
	CUDA_VISIBLE_DEVICES=$(AVAILABLE_GPUS) torchrun --nnodes=$(NUM_NODES) --nproc_per_node=$(NUM_GPUS) --master_port=$(MASTER_PORT_SR) -m scripts.train_sr --model DiT-XL/2 --image-size $(IMG_SIZE) --num-classes 1 --global-batch-size $(BATCH_SIZE) --ckpt-every $(CKPT_EVERY_N_ITERS) --log-every $(LOG_EVERY_N_ITERS) --results-dir $(CKPT_PATH_2) --data-path $(TRAINING_DSET_PATH_STAGE2) --pretrained_ckpt_path $(PRETRAINED_CKPT_PATH) --sr_scale $(SR_SCLAE)
#==================================================================================


#===Stage 2 sampling command=======================================================
sample_stage_2:
	python -m scripts.sample_sr --model DiT-XL/2 --image-size 512 --ckpt "your_ckpt_path_sr" --lr_path "your_low_res_path" --save_path "your_final_output_path" --num-sampling-steps 250