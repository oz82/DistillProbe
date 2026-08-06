#!/usr/bin/env bash
# Generate the per-file SHA-256 manifest that ships with the Data Availability
# (Zenodo) archive.
#
# The manifest is generated from the release archive itself (`git archive HEAD`,
# which applies the .gitattributes export-ignore rules) rather than from a
# hand-maintained find expression. A hand-maintained list drifts: the previous
# version certified only 98 of the 142 released files, so the manuscript's
# "per-file SHA-256 checksums" promise was not actually kept. Deriving the list
# from the archive makes the manifest cover exactly what is uploaded, by
# construction.
#
# Paths in the output are relative to the archive root, so a downloader can
# verify with:
#     sha256sum -c CHECKSUMS.sha256
# from the unpacked archive directory.
#
# Usage: bash scripts/gen_checksums.sh > CHECKSUMS.sha256
set -eu
cd "$(dirname "$0")/.."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git archive HEAD | tar -x -C "$tmp"

# CHECKSUMS.sha256 is itself a release artifact and is not part of the archive,
# so nothing needs to be excluded here: everything the archive contains gets a
# hash.
cd "$tmp"
find . -type f -printf '%P\n' \
  | LC_ALL=C sort \
  | xargs -d '\n' sha256sum
