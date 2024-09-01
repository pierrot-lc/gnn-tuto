platform:
  TF_CPP_MIN_LOG_LEVEL=0 python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"

tests:
  python3 -m pytest .

compile-requirements:
  uv pip freeze | uv pip compile - -o requirements.txt
pip-requirements:
  uv pip install -r requirements.txt
