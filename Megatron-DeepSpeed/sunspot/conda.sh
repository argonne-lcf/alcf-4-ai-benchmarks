#!/bin/bash
# This is the conda environment setup for Megatron-DeepSpeed code

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export PYTHONNOUSERSITE=1
export FILESYSTEM=/gila/
export PPN=12
export CC=icx
export CXX=icpx

export TRANSFER_PACKAGE=${TRANSER_PACKAGE:-0}
export BUILD=${BUILD:-"2024-08-29"}

if [ $TRANSFER_PACKAGE -eq 1 ]; then
    module use /soft/modulefiles
    module load frameworks
    export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)    
    export DST_DIR=/tmp/
    echo "Transfer built python package ($BUILD): `date`"
    # To remove completely the dependency on Lustre, one can copy the following files to /tmp and do transfer from there. 
    aprun --pmi=pmix -n $PBS_JOBSIZE -N 1 python ${FILESYSTEM}/Aurora_deployment/AuroraGPT/cache_soft.py \
	  --src ${FILESYSTEM}/Aurora_deployment/AuroraGPT/build/${BUILD}/soft.tar.gz \
	  --dst /tmp/soft.tar.gz --d
    
    aprun --pmi=pmix -n $PBS_JOBSIZE -N 1 python ${FILESYSTEM}/Aurora_deployment/AuroraGPT/cache_soft.py \
	  --src ${FILESYSTEM}/Aurora_deployment/AuroraGPT/build/${BUILD}/Megatron-DeepSpeed.tar.gz \
	  --dst /tmp/Megatron-DeepSpeed.tar.gz --d
    
    export MD=${DST_DIR}/Megatron-DeepSpeed/
    echo "Transfer built python package ($BUILD): `date` Done"
    # Other environment setup
    module unload frameworks
    module unload oneapi
    
    export PATH=$DST_DIR//anl_2024_q3-official_release/bin/:$PATH
    export PYTHONBASE=$DST_DIR//anl_2024_q3-official_release
    source $DST_DIR/ccl.sh
    which python
    module use /soft/preview/pe/24.180.0-RC4/modulefiles
    module use /opt/aurora/24.086.0/spack/gcc/0.7.0/modulefiles
    
    module add spack-pe-gcc/0.7.0-24.086.0.lua
    module add gcc/12.2.0
    
    module add oneapi/release/2024.2.1
    module add intel_compute_runtime/release/803.63.lua
    module add mpich/icc-all-pmix-gpu/20231026
    
    module use /soft/preview/gordonbell/graphics-compute-runtime/modulefiles
    module add graphics-compute-runtime/hotfix_agama-ci-devel-803.63
    
    python -m pip install /tmp/ezpz
    export AGPT_ROOT=${DST_DIR}/
    aprun --pmi=pmix -n ${PBS_JOBSIZE} -N 1 $AGPT_ROOT/soft/interposer.sh $DST_DIR/build_helper.sh    
else
    echo "Using package on lustre"
    export AGPT_ROOT=${FILESYSTEM}/Aurora_deployment/AuroraGPT/build/${BUILD}/
    export DST_DIR=${AGPT_ROOT}
    export PATH=$DST_DIR/anl_2024_q3-official_release/bin/:$PATH
    export PYTHONBASE=$DST_DIR/anl_2024_q3-official_release
    source $DST_DIR/ccl.sh
    module use /soft/preview/pe/24.180.0-RC4/modulefiles
    module use /opt/aurora/24.086.0/spack/gcc/0.7.0/modulefiles
    
    module add spack-pe-gcc/0.7.0-24.086.0.lua
    module add gcc/12.2.0
    
    module add oneapi/release/2024.2.1
    module add intel_compute_runtime/release/803.63.lua
    module add mpich/icc-all-pmix-gpu/20231026
    
    module use /soft/preview/gordonbell/graphics-compute-runtime/modulefiles
    module add graphics-compute-runtime/hotfix_agama-ci-devel-803.63
    export MD=${AGPT_ROOT}/Megatron-DeepSpeed
fi

echo "####### Python environment ###########"
which python
echo "--------------------------------------"

echo "####### Megatron-DeepSpeed ###########"
echo "Using Megatron-DeepSpeed code from $MD"
cd $MD
echo "Git commit: `git rev-parse HEAD`"
cd - 


IFS='.' read -ra ADDR <<< "$PBS_JOBID"
export JOBID=$ADDR
export PYTHONPATH=$MD:$PYTHONPATH
