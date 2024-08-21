platform:
  TF_CPP_MIN_LOG_LEVEL=0 python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"

download-dataset:
  wget https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip
  unzip NCI1.zip
  mkdir -p ./data
  mv NCI1 ./data
  rm NCI1.zip

compile-requirements:
  uv pip freeze | uv pip compile - -o requirements.txt
pip-requirements:
  uv pip install -r requirements.txt
