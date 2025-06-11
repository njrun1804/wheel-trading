#!/bin/bash
# Simple helper to stage all changes and create a git commit.

if [ -z "$1" ]; then
  echo "Usage: $0 \"commit message\"" >&2
  exit 1
fi

msg="$1"

# Stage all changes
git add -A

# Only commit if there is something to commit
if git diff --cached --quiet; then
  echo "Nothing to commit" >&2
  exit 0
fi

git commit -m "$msg"
