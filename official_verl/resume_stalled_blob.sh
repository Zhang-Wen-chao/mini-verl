#!/usr/bin/env bash
# Finish an image pull whose registry mirror stalled inside one large layer.
#
# Symptom: `nerdctl pull` parks at some fraction of a multi-GiB layer and stops
# making progress forever. The progress display keeps repainting and `elapsed`
# keeps climbing, so it looks alive; the tell is that the content store's disk
# usage stops growing. No error, no retry. Two different mirrors reproduced it.
#
# Why this works: a containerd content store is content-addressed. A file whose
# SHA-256 equals the layer digest is accepted as that layer, with no need to use
# containerd's own downloader. The partial bytes are already on disk under
# ingest/, so this resumes from them instead of re-fetching gigabytes.
#
# Usage:
#   DIGEST=sha256:<layer-digest> REPO=<registry-account>/<image> \
#     CONTAINERD_CONTENT_ROOT=/path/to/io.containerd.content.v1.content \
#     ./resume_stalled_blob.sh
#
# Afterwards, re-run the original pull: the layer resolves as "exists".
#
# This writes into the host-wide shared content store. The SHA-256 check below
# is the safety gate; do not remove it.
set -euo pipefail

: "${DIGEST:?Set DIGEST to the stalled layer digest, e.g. sha256:ba919f...}"
: "${REPO:?Set REPO to the image repository, e.g. verlai/verl}"
: "${CONTAINERD_CONTENT_ROOT:?Set CONTAINERD_CONTENT_ROOT to the content store root.}"
WORK=${WORK:?Set WORK to a scratch file path outside the content store.}

INGEST_DIR=$(find "$CONTAINERD_CONTENT_ROOT/ingest" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
if [[ -z "$INGEST_DIR" ]]; then
  echo "no ingest directory found under $CONTAINERD_CONTENT_ROOT/ingest" >&2
  echo "a pull must be in progress (or interrupted) for this to resume from" >&2
  exit 2
fi

echo "resuming from $INGEST_DIR/data"
mv "$INGEST_DIR/data" "$WORK"
rm -rf "$INGEST_DIR"

token() {
  curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
}

# --retry-all-errors is the point: a mirror that drops the connection silently
# is exactly the case curl's default retry policy declines to handle.
for attempt in $(seq 1 40); do
  if curl -L -C - --retry 5 --retry-delay 2 --retry-all-errors \
      --connect-timeout 20 --speed-time 60 --speed-limit 10240 \
      -H "Authorization: Bearer $(token)" -o "$WORK" \
      "https://registry-1.docker.io/v2/${REPO}/blobs/${DIGEST}"; then
    break
  fi
  echo "attempt $attempt failed; re-authenticating and resuming" >&2
done

actual=$(sha256sum "$WORK" | cut -d' ' -f1)
expected=${DIGEST#sha256:}
if [[ "$actual" != "$expected" ]]; then
  echo "SHA-256 mismatch: got $actual, expected $expected" >&2
  echo "refusing to install into the shared content store" >&2
  exit 1
fi

install -m 0644 "$WORK" "$CONTAINERD_CONTENT_ROOT/blobs/sha256/$expected"
echo "installed $expected; re-run the original pull"
