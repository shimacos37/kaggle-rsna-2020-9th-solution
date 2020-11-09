FROM rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04-py3.7

RUN apt-get update \
     && apt-get install --yes  --no-install-recommends \
     cuda-toolkit-10-2 \
     gcc \
     ssh-client \
     && apt-get clean

RUN ln -s /usr/local/cuda/bin/nvcc /opt/conda/envs/rapids/bin/ \
     && ln -s /usr/local/cuda/bin/nvcc.profile /opt/conda/envs/rapids/bin/ \
     && ln -s  /usr/local/cuda/nvvm /opt/conda/envs/rapids/
ENV CPATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$CPATH

RUN  source activate rapids \
     && pip install opencv-python \
     albumentations \
     tqdm \
     joblib \
     torch \
     torchvision \
     efficientnet-pytorch \
     pretrainedmodels \
     segmentation-models-pytorch \
     pytorch-lightning \
     hydra-core \
     hydra_colorlog \
     google-cloud-storage \
     seaborn \
     plotly \
     nilearn \
     h5py \
     wandb \
     jupyterlab \
     jupyterlab_code_formatter \
     black \
     mypy \
     lightgbm \
     flake8 \
     timm \
     xgboost \
     pydicom \
     rising \
     threadpoolctl \
     transformers \
     torchcontrib \
     torch_optimizer

RUN source activate rapids \
     && conda install gdcm -c conda-forge

WORKDIR /root/workdir