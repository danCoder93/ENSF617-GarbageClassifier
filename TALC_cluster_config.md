Follow the instructions


```sh
# Create env with Python 3.11 and proper channel order
conda create -y -n pytorch python=3.11 -c pytorch -c nvidia -c conda-forge
conda activate pytorch

# Enforce strict channel priority for this env
conda config --env --set channel_priority strict

# Install CUDA-enabled PyTorch + vision/audio + tensorboard
conda install -y pytorch torchvision torchaudio tensorboard transformers pytorch-cuda=12.1 \
  -c pytorch -c nvidia -c conda-forge --override-channels
ls
# Quick sanity check (compile-time CUDA)
python -c "import torch; print('Torch:', torch.__version__, '| CUDA runtime:', torch.version.cuda)"

# --- Jump into GPU allocation and verify GPU visibility ---
srun -p gpu --gres=gpu:1 --pty bash -c "
source ~/software/init-conda 2>/dev/null
conda activate pytorch
echo 'Host:' \$(hostname)
echo 'CUDA_VISIBLE_DEVICES=' \$CUDA_VISIBLE_DEVICES
nvidia-smi -L
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA built:', torch.backends.cuda.is_built())
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
"

```
