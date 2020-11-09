for i in $(seq 0 4); do
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
    python src/rincha/main.py exp.fold=$i
done

for i in $(seq 0 4); do
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
    python src/rincha/main.py exp.fold=$i \
    model.figsize=512 \
    model.name=tf_efficientnet_b5_ns \
    model.batch_size=16 \
    exp.name=512-b5
done
