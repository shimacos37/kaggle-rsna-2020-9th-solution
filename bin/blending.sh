
docker run --rm -it \
  -v $PWD/:/root/workdir/ \
  kaggle/pytorch:rsna \
  python src/shimacos/scripts/blending.py
