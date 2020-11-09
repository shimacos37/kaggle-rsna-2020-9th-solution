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
        python src/shimacos/main_nn.py \
            base.opt_name=adam \
            base.loss_name=rsna_split_loss \
            data.n_fold=$n_fold \
            data.is_train=True \
            data.dataset_name=feature_dataset \
            data.image_mode=concat \
            train.learning_rate=0.00005 \
            store.model_name=concat_512_384_cnn \
            model.backbone=cnn \
            model.model_name=deconv_feature_model \
            model.dropout_rate=0.2 \
            model.num_feature=3591 \
            train.batch_size=64 \
            train.epoch=50 \
            train.patience=10 \
            test.batch_size=128 \
            test.is_validation=True
done

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
        python src/shimacos/main_nn.py \
            base.opt_name=adam \
            base.loss_name=rsna_split_loss \
            data.n_fold=$n_fold \
            data.is_train=True \
            data.dataset_name=feature_dataset \
            data.image_mode=concat \
            train.learning_rate=0.00005 \
            store.model_name=concat_512_384_gru \
            model.backbone=gru \
            model.model_name=deconv_feature_model \
            model.dropout_rate=0.2 \
            model.num_feature=3591 \
            train.batch_size=64 \
            train.epoch=50 \
            train.patience=10 \
            test.batch_size=128 \
            test.is_validation=True
done


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
        python src/shimacos/main_nn.py \
            base.opt_name=adam \
            base.loss_name=rsna_split_loss \
            data.n_fold=$n_fold \
            data.is_train=True \
            data.dataset_name=feature_dataset \
            data.image_mode=concat \
            train.learning_rate=0.00005 \
            store.model_name=concat_512_384_lstm \
            model.backbone=lstm \
            model.model_name=deconv_feature_model \
            model.dropout_rate=0.2 \
            model.num_feature=3591 \
            train.batch_size=64 \
            train.epoch=50 \
            train.patience=10 \
            test.batch_size=128 \
            test.is_validation=True
done


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
        python src/shimacos/main_nn.py \
            base.opt_name=adam \
            base.loss_name=rsna_split_loss \
            data.n_fold=$n_fold \
            data.is_train=True \
            data.dataset_name=feature_dataset \
            data.image_mode=concat \
            train.learning_rate=0.00005 \
            store.model_name=concat_512_384_cnn_lstm \
            model.backbone=cnn_rnn \
            model.model_name=deconv_feature_model \
            model.dropout_rate=0.2 \
            model.num_feature=3591 \
            train.batch_size=64 \
            train.epoch=50 \
            train.patience=10 \
            test.batch_size=128 \
            test.is_validation=True
done
