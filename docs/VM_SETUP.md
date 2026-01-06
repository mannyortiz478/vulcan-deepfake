# Google Cloud VM (GPU) setup notes

Quick checklist for getting the repo running on a GCP VM with GPU:

1. Create VM with a recent NVIDIA driver and a CUDA-enabled image (Ubuntu 22.04 recommended). Example using gcloud:

```bash
gcloud compute instances create gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud
```

2. SSH into VM and install NVIDIA drivers and CUDA if not preinstalled. If using Deep Learning images this is mostly preconfigured.

3. Create Python environment and install PyTorch matching CUDA:

```bash
# recommended inside VM
python -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip
# pick correct wheel for your CUDA version; example for CUDA 11.8:
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchaudio
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install transformers
```

4. Verify GPU accessibility:

```bash
python -c "import torch; print('cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

5. Run a smoke test / small training run (example):

```bash
python train_and_test.py --asv_path /path/to/asv --config configs/training/whisper_specrnet.yaml --device cuda --amp --batch_size 8 --epochs 1
```

6. Example run helper

We include `examples/run_gce.sh` that launches training in background and redirects logs to a file. Example:

```bash
# Launch a train with amp and accumulation 2 and tqdm enabled
./examples/run_gce.sh /path/to/asv configs/training/whisper_specrnet.yaml cuda 8 10 2 --amp --use-tqdm mytrain.log
# Then tail logs
tail -f mytrain.log
```

7. Systemd service example

You can create a systemd service to manage long-running training jobs. An example service is included at `examples/gce_train.service`. Copy it to `/etc/systemd/system/` and update paths and user, then enable and start it:

```bash
sudo cp examples/gce_train.service /etc/systemd/system/gce_train.service
sudo systemctl daemon-reload
sudo systemctl enable --now gce_train.service
sudo journalctl -u gce_train.service -f
```

Notes & tips:

- If you run out of GPU memory, reduce `--batch_size` or use `--accum-steps` to effectively increase batch size without raising memory.
- Use `--amp` to enable mixed precision for faster training and lower memory usage on GPUs.
- If you use multiple GPUs you can set `--device cuda:0` or similar; distributed training is not included in this repo yet.
