# Deploying the federation console

Runs the dashboard **and** the training in one container, so the live console
streams real federations rather than only showing stored results. Training and
the dashboard must share a filesystem — the console works by tailing an
append-only event log — so splitting them across two hosts leaves the live view
permanently empty.

## Requirements

| | |
|---|---|
| RAM | **4 GB minimum, 6–8 GB comfortable.** Each of the 5 sites is a Ray worker holding its own shard plus torch. Below ~3.5 GB workers are OOM-killed mid-round, which shows up as a run that quietly stops updating rather than an error. |
| Disk | ~4 GB (2.5 GB image, ~400 MB dataset, results) |
| CPU | 2+ cores. A 10-round run takes ~5 min on a laptop and noticeably longer on a shared-CPU VM. |

Free tiers (typically 512 MB) cannot run training. They can serve the stored
results read-only — set `ALLOW_LAUNCH=false`.

## Quick start

```bash
git clone <your repo> && cd federated_learning_system

cat > .env <<'EOF'
DASHBOARD_USER=admin
DASHBOARD_PASSWORD=<a long random password>
ALLOW_LAUNCH=true
EOF

docker compose up -d --build
docker compose logs -f          # first boot downloads ~200 MB
```

Then browse to `http://127.0.0.1:8000` and sign in.

The first boot downloads the dataset from UCI and rebuilds `data_pruned.csv`
(dropping the 267 zero-variance genes, exactly as the EDA notebook does). It
lands in a named volume, so restarts and rebuilds reuse it.

## Exposing it publicly

Compose binds to `127.0.0.1` on purpose. **Do not change that to `0.0.0.0`
without TLS in front** — basic-auth credentials would cross the network in
cleartext. Two good options:

**Cloudflare Tunnel** — no open ports, no certificates, works behind NAT:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

**Caddy** — automatic Let's Encrypt certificates, if you have a domain:

```
console.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

`serve.py` runs with `proxy_headers`, so client IPs and scheme are read from the
proxy correctly.

## Security posture

- **Authentication is mandatory when launching is enabled on a public address.**
  The container refuses to start otherwise, rather than quietly exposing a
  machine that spawns CPU-heavy jobs and rewrites the Flower config. That check
  is in `serve.py`, not just the entrypoint, so it holds however you start it.
- Every route requires credentials except `/api/health`, which stays open for
  container health probes and carries no data.
- Credentials are compared with `hmac.compare_digest`; a plain `==` would leak
  the password's length and prefix through timing.
- `ALLOW_LAUNCH=false` disables starting runs entirely — the four stored-results
  views still work. Use it for anything long-lived or widely shared.
- **No patient data is involved.** The expression matrix never enters the
  dashboard; it serves aggregate metrics, label histograms and gene identifiers,
  and the UCI data is public.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DASHBOARD_PASSWORD` | — | Required when `ALLOW_LAUNCH=true`. No default by design. |
| `DASHBOARD_USER` | `admin` | |
| `ALLOW_LAUNCH` | `true` | `false` = read-only |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | Inside the container |
| `DATA_ROOT` | `/data` | Dataset volume |

## Operating it

```bash
docker compose logs -f console          # follow
docker compose restart console          # restart (dataset is kept)
docker compose down                     # stop, keep volumes
docker compose down -v                  # also delete the dataset volume
docker stats fl-rnaseq-console          # watch memory during a run
```

Results are bind-mounted to `./fl_rnaseq/results`, so runs started in the
container appear in your working tree and can be committed.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Container exits with code 2 immediately | `ALLOW_LAUNCH=true` without `DASHBOARD_PASSWORD`. The logs say so explicitly. |
| Run starts, then the global model stops changing | Workers OOM-killed. Raise the memory limit or reduce sites. The results file records `health.degraded`, and the console shows a red banner rather than passing it off as slow convergence. |
| First boot takes several minutes | The ~200 MB UCI download plus the pruning pass. Only happens once per volume. |
| `Start run` greyed out | A run is already in progress — only one at a time, deliberately. Use **Stop run**. |
| Ray warnings about `/dev/shm` | Raise `shm_size` in compose; the 64 MB Docker default makes Ray spill to disk. |
