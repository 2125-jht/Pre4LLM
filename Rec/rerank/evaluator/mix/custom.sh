export ENABLE_MIO_TENSORFLOW_GPU=true
export ENABLE_COMMON_LEAF_PYTHON_WRAPPER=false
export COMMON_RECO_LEAF_EXTRA_PROCESSORS="mio mio_rpc kuiba offline colossus_client kap gsu cofea tdm"
export COMMON_RECO_LEAF_LINK_MKL=true
export SIMPLE_MIO_BUILD_LIBTRAINER_OP=true
#export ENABLE_GCC8=false
export CUDA_DIR=/data/soft/gpu10.1/cuda
export BUILD_TF_WITH_CUDA_VERSION=10.1
export ENABLE_KAI_LEARNER=true
if [[ -n $REPO_URL ]] && [[ "${USE_KAI_GIT_REPO}" == "true" ]]; then
    pushd ${JOB_HOST_DIR}
    mv teams/aiplatform/kai teams/aiplatform/kai.svn
    ln -srf ${LOCAL_GIT_DIRECTORY}/kai teams/aiplatform/
    export GIT_SUFFIX=true
    popd
fi
