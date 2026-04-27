---
name: resolve-review
description: Use when a LMDeploy pull request has review comments to address. Verify repo state, branch, env, and `gh` auth first; then fetch comments, make minimal fixes, validate locally, and commit/push only when the branch is ready.
---

# Resolve PR Review Comments

## 1. Preflight checks

Before fetching or editing anything, verify:

- You are inside the intended LMDeploy repo.
- `gh auth status` succeeds.
- The current branch is the PR branch you intend to update.
- `git status --short` is understood. Do not overwrite unrelated local changes.
- The correct Python env is available for lint/tests.

`<env>` = `fp8` for `lmdeploy_fp8`, `vl` for `lmdeploy_vl`.

Useful checks:

```bash
git remote -v
git branch --show-current
git status --short
gh auth status
```

If the tree is dirty:

- Continue only if the edits are part of the PR work.
- Otherwise avoid touching them and stage only your files.

## 2. Fetch comments

Prefer `gh pr view <PR>` first to confirm the PR title, branch, and overall state.

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

Prioritize unresolved, actionable comments.

## 3. Fix each issue

For each comment:

- Read the flagged file and surrounding logic, not just the single line.
- Confirm whether the comment is still applicable on the current branch tip.
- Make the smallest change that resolves the concern.
- Keep a short running list of: comment, affected file, and fix summary.

## 4. Validate locally

Run the narrowest useful validation first.

- Prefer targeted tests for the touched files or subsystem.
- Run `pre-commit` if repo policy expects it.
- If a comment changes behavior, include a reproducer or test when practical.

```bash
pre-commit run --all-files
```

## 5. Stage & commit

```bash
git add <fixed files>
git commit -m "fix: address PR review comments"
```

Only commit if:

- The intended files are staged.
- Validation results are known.
- The commit does not include unrelated changes.

Check `git status` before committing.

## 6. Push

```bash
git push
```

Push only after confirming you are updating the correct remote branch.

## Output Contract

This skill produces:

- List of comments addressed (file path + line + summary of fix)
- Validation status (targeted tests and/or pre-commit pass/fail)
- Commit SHA of the fix commit
- Push confirmation (remote branch updated)
