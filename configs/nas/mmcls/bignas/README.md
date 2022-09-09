



GPUS=1 GPUS_PER_NODE=1 tools/slurm_test.sh pat_dev xxx configs/nas/mmcls/autoformer/autoformer_supernet_8xb256_in1k.py /mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/.base/best_accuracy_top-1_epoch_297_mmrazor_r1.pth

GPUS=1 GPUS_PER_NODE=1 tools/slurm_test.sh pat_dev xxx configs/nas/bignas/bignas_mobilenetv3_large_supernet_32xb64.py


/mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.pth

/mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/.base/gml/configs/nas/bignas/bignas_mobilenetv3_large_subnet_16xb128_flops600M.py


GPUS=1 GPUS_PER_NODE=1 tools/slurm_test.sh pat_dev xxx configs/nas/bignas/bignas_mobilenetv3_large_subnet_16xb128_flops600M.py /mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.pth --work-dir work_dirs/0916_test --eval accuracy --cfg-options algorithm.mutable_cfg=/mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.yaml

SRUN_ARGS='-p pat_dev -x HOST-10-198-32-[12,22]' GPUS=16 tools/slurm_test.sh pat_dev xxx configs/nas/bignas/bignas_mobilenetv3_large_subnet_16xb128_flops600M.py /mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.pth --work-dir work_dirs/0916_test --eval accuracy --cfg-options algorithm.mutable_cfg=/mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.yaml

SRUN_ARGS='-p pat_dev -x HOST-10-198-32-[12,22]' GPUS=8 tools/slurm_test.sh pat_dev xxx configs/nas/bignas/bignas_mobilenetv3_large_subnet_16xb128_flops600M.py /mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.pth --work-dir work_dirs/0916_test --eval accuracy --cfg-options algorithm.mutable_cfg=/mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.yaml

SRUN_ARGS='-p pat_dev -x HOST-10-198-32-[12,22]' GPUS=8 tools/slurm_test.sh pat_dev xxx configs/nas/bignas/bignas_mobilenetv3_large_subnet_16xb128_flops600M.py /mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.pth --work-dir work_dirs/0916_test --eval accuracy --cfg-options algorithm.mutable_cfg=/mnt/lustre/sunyue1/final_subnet_step1200_20220320_2134.yaml

SRUN_ARGS='-p pat_dev -x HOST-10-198-32-[12,22]' GPUS=8 tools/slurm_test.sh pat_dev xxx configs/nas/bignas/bignas_mobilenetv3_large_subnet_16xb128_flops600M.py /mnt/lustre/sunyue1/epoch_360.pth --work-dir work_dirs/0916_test --eval accuracy --cfg-options algorithm.mutable_cfg=/mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/.base/model_convert/bignas_model/supernet.yaml
