#!/usr/bin/env bash
T=`date +%m%d%H%M`

MMDET3D=${MMDET3D:-/mnt/cache/huangbin1/m2bev}
SRUN_ARGS=${SRUN_ARGS:-"-s"}
RETRY=${RETRY:-1}
echo MMDET3D: $MMDET3D
echo SRUN_ARGS: $SRUN_ARGS
echo RETRY: $RETRY
START_TIME=`date +%Y%m%d-%H:%M:%S`


function slurm_train {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=${4:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
    LOG_FILE=logs/paper_train_${START_TIME}.log

    echo slurm_train; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE \
    sh ./tools/slurm_train.sh \
        $PARTITION \
        $JOB_NAME \
        configs/fastbev/exp/$EXPNAME.py \
        work_dirs/fastbev/exp/$EXPNAME \
        --resume-from work_dirs/fastbev/exp/$EXPNAME/latest.pth \
    2>&1 | tee $LOG_FILE > /dev/null &
}

function slurm_test {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=${4:-`basename $EXPNAME`}
    RESUME=${5:-work_dirs/fastbev/exp/$EXPNAME/latest.pth}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo slurm_test; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE QUOTATYPE=$QUOTATYPE \
    sh ./tools/slurm_test.sh \
        $PARTITION \
        $JOB_NAME \
        configs/fastbev/exp/$EXPNAME.py \
        $RESUME \
        --out work_dirs/fastbev/exp/$EXPNAME/results/results.pkl \
        --format-only \
        --eval-options jsonfile_prefix=work_dirs/fastbev/exp/$EXPNAME/results \
        2>&1 | tee work_dirs/fastbev/exp/$EXPNAME/log.test.$JOB_NAME.$T.txt > /dev/null &
}

function slurm_eval {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=${4:-`basename $EXPNAME`}
    RESULT=${5:-work_dirs/fastbev/exp/$EXPNAME/results/results.pkl}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo slurm_eval; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE QUOTATYPE=$QUOTATYPE \
    sh ./tools/slurm_eval.sh \
        $PARTITION \
        $JOB_NAME \
        configs/fastbev/exp/$EXPNAME.py \
        --out $RESULT \
        --eval bbox \
        2>&1 | tee work_dirs/fastbev/exp/$EXPNAME/log.eval.$JOB_NAME.$T.txt > /dev/null &
}

function batch_slurm_test {
    PARTITION=$1
    GPUS=$2
    EXPNAME=$3
    JOB_NAME=${4:-`basename $EXPNAME`}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
    mkdir -p work_dirs/fastbev/exp/$EXPNAME/batch
    mkdir -p work_dirs/fastbev/exp/$EXPNAME/test/
    for RESUME in `ls work_dirs/fastbev/exp/$EXPNAME/batch/epoch*.pth`; do
        BASENAME=`basename $RESUME`
        SUB_JOB_NAME=test-$JOB_NAME-$BASENAME
        echo slurm_test; sleep 0.5s
        echo RESUME: $RESUME
        echo SUB_JOB_NAME: $SUB_JOB_NAME
        echo LOG: work_dirs/fastbev/exp/$EXPNAME/test/log.test.$BASENAME.$T
        MMDET3D=$MMDET3D \
        SRUN_ARGS=$SRUN_ARGS \
        GPUS=$GPUS GPUS_PER_NODE=$GPUS_PER_NODE QUOTATYPE=$QUOTATYPE \
        sh ./tools/slurm_test.sh \
            $PARTITION \
            $SUB_JOB_NAME \
            configs/fastbev/exp/$EXPNAME.py \
            $RESUME \
            --out work_dirs/fastbev/exp/$EXPNAME/test/$BASENAME/results.$BASENAME.pkl \
            --eval bbox \
            --eval-options jsonfile_prefix=work_dirs/fastbev/exp/$EXPNAME/test/$BASENAME \
            2>&1 | tee work_dirs/fastbev/exp/$EXPNAME/test/log.test.$BASENAME.$T \
        &
    done
}

function slurm_benchmark {
    PARTITION=$1
    EXPNAME=$2
    JOB_NAME=benchmark-${3:-`basename $EXPNAME`}

    echo slurm_analysis; sleep 0.5s
    MMDET3D=$MMDET3D \
    SRUN_ARGS=$SRUN_ARGS \
    GPUS=1 GPUS_PER_NODE=1 \
    sh ./tools/slurm_analysis.sh \
        $PARTITION \
        $JOB_NAME \
        configs/fastbev/exp/$EXPNAME.py \
        work_dirs/fastbev/exp/$EXPNAME/latest.pth
}


PARTITION=$1
QUOTATYPE=$2

# train

# M0-5
# slurm_train $PARTITION 32 paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4
# slurm_train $PARTITION 32 paper/fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4
# slurm_train $PARTITION 32 paper/fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4
# slurm_train $PARTITION 32 paper/fastbev_m3_r34_s256x704_v200x200x6_c256_d6_f4
# slurm_train $PARTITION 32 paper/fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4
# slurm_train $PARTITION 32 paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4

# Multi-scale & ablation
# slurm_train $PARTITION 32 paper/fastbev_r50_v1_woms_e48_s256x704_v200x200x4_c256_d2_f4
# slurm_train $PARTITION 32 paper/fastbev_r50_v2_imgms_e48_s256x704_v200x200x4_c256_d2_f4
# slurm_train $PARTITION 32 paper/fastbev_r50_v3_bevms_e48_s256x704_v200x200x4_c256_d2_f4
# slurm_train $PARTITION 32 paper/fastbev_r50_v4_ms_e48_s256x704_v200x200x4_c256_d2_f4
# slurm_train $PARTITION 32 paper/fastbev_r50_v4_ms_20cbgs_s256x704_v200x200x6_c256_d6_f4

# test
# slurm_test $PARTITION 16 paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4

# eval
# slurm_eval $PARTITION 1 paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4
