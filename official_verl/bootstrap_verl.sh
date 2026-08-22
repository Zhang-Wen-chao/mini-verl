#!/usr/bin/env bash
# Fetch one explicit official-verl revision on the Linux CUDA training host.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "official verl training must be bootstrapped on a Linux CUDA host" >&2
  exit 2
fi

target_dir=${1:-.official-verl/verl}
revision=${VERL_REVISION:?Set VERL_REVISION to an official release tag or full commit SHA.}

if [[ -e "$target_dir" && ! -d "$target_dir/.git" ]]; then
  echo "refusing to use non-git target: $target_dir" >&2
  exit 2
fi

if [[ ! -d "$target_dir/.git" ]]; then
  git clone https://github.com/volcengine/verl.git "$target_dir"
fi

git -C "$target_dir" fetch --tags --prune origin
git -C "$target_dir" checkout --detach "$revision"
printf 'Official verl pinned at %s\n' "$(git -C "$target_dir" rev-parse HEAD)"
printf 'Next: python official_verl/preflight.py --verl-dir %s --require-cuda --write-lock\n' "$target_dir"
