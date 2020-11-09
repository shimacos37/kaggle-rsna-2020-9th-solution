
# LGBM
docker run --rm -it \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    -v $HOME/.git/:/root/.git \
    -e SLURM_LOCALID=0 \
    --shm-size=600gb \
    --ipc=host \
    --security-opt seccomp=unconfined \
    kaggle/pytorch:rsna \
    python ./src/shimacos/stacking_lgbm.py store.model_name=stacking_lgbm

# CNN
for n_fold in $(seq 0 4);
do
    docker run --rm -it \
        -v $PWD/:/root/workdir/ \
        -v $HOME/.config/:/root/.config \
        -v $HOME/.netrc/:/root/.netrc \
        -v $HOME/.cache/:/root/.cache \
        -v $HOME/.git/:/root/.git \
        -e SLURM_LOCALID=0 \
        --runtime=nvidia \
        --shm-size=600gb \
        --ipc=host \
        --security-opt seccomp=unconfined \
        kaggle/pytorch:rsna \
        python ./src/shimacos/main_nn.py \
            base.opt_name=adam \
            base.loss_name=rsna_split_loss \
            data.n_fold=$n_fold \
            data.is_train=True \
            data.dataset_name=stacking_dataset \
            train.learning_rate=0.0001 \
            store.model_name=stacking_cnn \
            model.backbone=cnn \
            model.model_name=stacking_model \
            model.dropout_rate=0.2 \
            train.batch_size=128 \
            train.epoch=50 \
            train.patience=10 \
            test.batch_size=128 
done

# GRU
for n_fold in $(seq 0 4);
do
    docker run --rm -it \
        -v $PWD/:/root/workdir/ \
        -v $HOME/.config/:/root/.config \
        -v $HOME/.netrc/:/root/.netrc \
        -v $HOME/.cache/:/root/.cache \
        -v $HOME/.git/:/root/.git \
        -e SLURM_LOCALID=0 \
        --runtime=nvidia \
        --shm-size=600gb \
        --ipc=host \
        --security-opt seccomp=unconfined \
        kaggle/pytorch:rsna \
        python ./src/shimacos/main_nn.py \
            base.opt_name=adam \
            base.loss_name=rsna_split_loss \
            data.n_fold=$n_fold \
            data.is_train=True \
            data.dataset_name=stacking_dataset \
            train.learning_rate=0.0001 \
            store.model_name=stacking_gru \
            model.backbone=gru \
            model.model_name=stacking_model \
            model.dropout_rate=0.2 \
            train.batch_size=128 \
            train.epoch=50 \
            train.patience=10 \
            test.batch_size=128 
done