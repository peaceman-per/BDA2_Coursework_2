#!/usr/bin/env bash
# capture_submission_evidence.sh
# Run this script from the project root immediately before creating the
# submission zip.  It writes a timestamped directory listing to evidence/ls-l.txt
# which can be included in the zip to satisfy the "unaltered submission" requirement.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVIDENCE_DIR="${PROJECT_ROOT}/evidence"

mkdir -p "${EVIDENCE_DIR}"

OUTFILE="${EVIDENCE_DIR}/ls-l.txt"

echo "Capturing directory listing to ${OUTFILE} …"
{
    echo "=== Submission evidence: directory listing ==="
    echo "=== Captured at: $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
    echo ""
    ls -lR "${PROJECT_ROOT}"
} > "${OUTFILE}"

echo "Done.  Include evidence/ls-l.txt in your submission zip."
