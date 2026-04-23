---
name: resolve-review
description: Use when a PR has inline or top-level review comments to address — fetches all comments via gh api, guides through fixing each, then lints, commits, and pushes.
---

# Resolve PR Review Comments

## 1. Fetch comments

`<env>` = `fp8` for lmdeploy_fp8, `vl` for lmdeploy_vl.

Inline review comments (on specific lines):

```bash
conda run -n <env> gh api repos/InternLM/lmdeploy/pulls/<PR>/comments \
  | python3 -c "
import json, sys
for c in json.load(sys.stdin):
    print(f'[{c[\"path\"]}:{c.get(\"line\",\"?\")}]')
    print(c['body'])
    print()
"
```

Top-level review body comments (not attached to a line):

```bash
conda run -n <env> gh api repos/InternLM/lmdeploy/pulls/<PR>/reviews \
  | python3 -c "
import json, sys
for r in json.load(sys.stdin):
    if r.get('body'):
        print(f'[review by {r[\"user\"][\"login\"]}]')
        print(r['body'])
        print()
"
```

## 2. Fix each issue

Read the flagged file, understand the comment, edit the file.

## 3. Lint

```bash
pre-commit run --all-files
```

## 4. Stage & commit

```bash
git add <fixed files>
git commit -m "fix: address PR review comments"
```

## 5. Push

```bash
git push
```

## Output Contract

This skill produces:

- List of comments addressed (file path + line + summary of fix)
- Lint status (pre-commit pass/fail)
- Commit SHA of the fix commit
- Push confirmation (remote branch updated)
