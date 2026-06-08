# TPU 2026 — Student Access Guide

Your team has a **dedicated Google Cloud TPU** (v6e-1) already running. These are
provisioned for **7 days** — you do **not** need to create, start, or stop
anything. Just connect and work.

> ⚠️ **Do NOT switch the TPU off.** Don't run `sudo poweroff` / `shutdown` /
> `halt`, and don't try to delete it. The machines are managed for you. If a VM
> goes down it has to be re-provisioned by hand and your whole team loses access
> until then. There is **no reason** to ever turn it off.

---

## 1. One-time setup on your laptop

1. **Install the Google Cloud CLI** (`gcloud`): https://cloud.google.com/sdk/docs/install
   - macOS: `brew install --cask google-cloud-sdk`
2. **Log in with the Gmail you registered** (the one in the class group):
   ```bash
   gcloud auth login
   gcloud config set project tpu-2026
   ```

That's it — no SSH keys to manage, no IPs to configure. Access is granted to your
Gmail through the class Google Group.

---

## 2. Find your team's machine

The VM name is your **Team ID**:

| Team | VM name | Team | VM name |
|------|---------|------|---------|
| aldimu | `aldimu` | dakolo | `dakolo` |
| waxvhe | `waxvhe` | chfazhvi | `chfazhvi` |
| begrla | `begrla` | ya | `ya-team` |
| fugupf | `fugupf` | giab | `giab` |
| fajizh | `fajizh` | chlina | `chlina` |
| depexu | `depexu` | chmawa | `chmawa` |
| hamcpaso | `hamcpaso` | | |

> Note: the `ya` team's machine is called **`ya-team`** (names must be ≥3 characters).

Set it once per terminal:
```bash
export TEAM=<your-vm-name>      # e.g. export TEAM=fugupf
```

---

## 3. Connect (SSH)

The VMs have no public IP, so you connect through Google's **IAP tunnel**
(this needs the `alpha` component — gcloud will offer to install it the first time):

```bash
gcloud alpha compute tpus tpu-vm ssh $TEAM \
  --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
```

The first connection takes ~30s while your key propagates. You land in a shared
Linux home directory on the TPU VM. You have **sudo**, so you can install anything
(`sudo apt install ...`, `pip install ...`, etc.).

---

## 4. Set up the Python / JAX / tunix environment

Everything you need is in this repo. On the VM:

```bash
git clone https://github.com/borisbolliet/tpu-2026.git
cd tpu-2026
./bootstrap.sh          # creates the venv and installs jax + tunix + flax + libtpu
```

Verify the TPU is visible:
```bash
source ~/venvs/tunix/bin/activate
python -c "import jax; print(jax.default_backend(), jax.devices())"
# expect: tpu [TpuDevice(...)]
```

Full details of what `bootstrap.sh` does (and how to do it by hand) are in
**[`tpu-setup.md`](tpu-setup.md)**.

---

## 5. JupyterLab

Two terminals on your laptop.

**Terminal 1** — open the tunnel (leave it running):
```bash
gcloud alpha compute tpus tpu-vm ssh $TEAM --zone=us-east5-a --project=tpu-2026 \
  --tunnel-through-iap -- -L 8888:localhost:8888 -N
```

**Terminal 2** — SSH in and launch Jupyter on the TPU:
```bash
gcloud alpha compute tpus tpu-vm ssh $TEAM --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap
# then on the VM:
source ~/venvs/tunix/bin/activate
jupyter lab --no-browser --port=8888 --ip=127.0.0.1
```

Open the printed `http://127.0.0.1:8888/lab?token=...` URL in your laptop browser,
and pick the **tunix** kernel. (TensorBoard works the same way on port 6006 — see
`tpu-setup.md`.)

---

## 6. Save your work — the VM is temporary

The TPUs **auto-delete after 7 days**, and anything left only on the VM is gone
when that happens. **Push your code to GitHub** (or copy results off the VM)
regularly. See the "Pushing changes back from the TPU VM" section of
[`tpu-setup.md`](tpu-setup.md) for the git-over-HTTPS + token recipe.

---

## Questions?

Ask in the **lab channel on Discord**.
