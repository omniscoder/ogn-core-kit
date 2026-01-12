#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/public_audit.sh [options]

Runs checks that help keep this repo safe to make public:
  - fail on tracked symlinks (public clone footgun)
  - fail on known local-machine path leaks in tracked files
  - run gitleaks using .gitleaks.toml

Options:
  --log-opts STR         Pass through to `gitleaks detect --log-opts` (scan a commit range).
  --no-git               Pass through to `gitleaks detect --no-git` (scan working tree only).
  --gitleaks-version V   Override gitleaks version (default: 8.30.0).
  -h, --help             Show this message.
EOF
}

GITLEAKS_VERSION="${GITLEAKS_VERSION:-8.30.0}"
GITLEAKS_LOG_OPTS=""
GITLEAKS_NO_GIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-opts)
      shift
      GITLEAKS_LOG_OPTS="${1:?--log-opts requires a string}"
      ;;
    --no-git)
      GITLEAKS_NO_GIT=1
      ;;
    --gitleaks-version)
      shift
      GITLEAKS_VERSION="${1:?--gitleaks-version requires a version}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[public-audit] error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift || true
done

if [[ "${GITLEAKS_NO_GIT}" == "1" && -n "${GITLEAKS_LOG_OPTS}" ]]; then
  echo "[public-audit] error: --no-git and --log-opts are mutually exclusive" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

die() {
  echo "[public-audit] error: $*" >&2
  exit 1
}

note() {
  echo "[public-audit] $*" >&2
}

require_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

download_gitleaks() {
  local out_dir="$1"
  local tgz="gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz"
  local url="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/${tgz}"

  note "downloading gitleaks v${GITLEAKS_VERSION} ..."
  curl -fsSL -o "${out_dir}/${tgz}" "$url"
  tar -xzf "${out_dir}/${tgz}" -C "${out_dir}"
  [[ -x "${out_dir}/gitleaks" ]] || die "gitleaks binary not found after extract"
  echo "${out_dir}/gitleaks"
}

run_gitleaks() {
  require_file ".gitleaks.toml"

  local gitleaks_bin=""
  if command -v gitleaks >/dev/null 2>&1; then
    gitleaks_bin="gitleaks"
  else
    local tmp
    tmp="$(mktemp -d)"
    gitleaks_bin="$(download_gitleaks "$tmp")"
  fi

  declare -a args=(detect --source . --redact --no-banner --config .gitleaks.toml)
  if [[ "${GITLEAKS_NO_GIT}" == "1" ]]; then
    args+=(--no-git)
  elif [[ -n "${GITLEAKS_LOG_OPTS}" ]]; then
    args+=(--log-opts "${GITLEAKS_LOG_OPTS}")
  fi

  note "running gitleaks ..."
  "$gitleaks_bin" "${args[@]}"
  note "gitleaks: OK"
}

check_tracked_symlinks() {
  local symlinks
  symlinks="$(git ls-files --stage | awk '$1=="120000"{print $4}')"
  if [[ -n "$symlinks" ]]; then
    echo "$symlinks" | sed 's/^/[public-audit] tracked symlink: /' >&2
    die "tracked symlinks found (remove before making repo public)"
  fi
  note "tracked symlinks: none"
}

check_known_local_paths() {
  local hits=""
  hits="$(git grep -nE '/home/chris|/mnt/d/' -- . ':(exclude)tools/public_audit.sh' ':(exclude)public_audit.sh' || true)"
  if [[ -n "$hits" ]]; then
    echo "$hits" >&2
    die "found local-machine paths in tracked files"
  fi
  note "local-machine path scan: OK"
}

main() {
  check_tracked_symlinks
  check_known_local_paths
  run_gitleaks
  note "public audit passed"
}

main "$@"

