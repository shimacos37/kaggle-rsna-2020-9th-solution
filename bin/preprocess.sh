
# Extract metadata
docker run --rm -it \
  -v $PWD/:/root/workdir/ \
  kaggle/pytorch:rsna \
  python src/shimacos/scripts/extract_metadata.py

# Clean data and add fold
docker run --rm -it \
  -v $PWD/:/root/workdir/ \
  kaggle/pytorch:rsna \
  python src/shimacos/scripts/clean_data.py
