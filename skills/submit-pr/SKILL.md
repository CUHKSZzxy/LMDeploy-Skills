---
name: submit-pr
description: Use when local LMDeploy changes are ready to become a pull request. Verify repo state, base branch, `gh` auth, and validation first; then stage only intended files, commit, push, and create the PR.
---

# Submit a PR for LMDeploy

## 1. Preflight checks

Before changing branches or pushing anything, verify:

- You are in the intended LMDeploy repo and remote points to `InternLM/lmdeploy`.
- `gh auth status` succeeds.
- `git status --short` is understood and does not contain accidental changes.
- The default base branch name is known (`main` in most cases, but do not assume blindly).
- Validation requirements for this change are known.

Useful checks:

```bash
git remote -v
git branch --show-current
git status --short
gh auth status
git symbolic-ref refs/remotes/origin/HEAD
```

If unrelated local changes exist, keep them out of the PR.

## 2. Create or confirm branch

Skip this step if already on a feature branch.

```bash
git switch <base-branch>
git pull --ff-only
git switch -c <type>/<short-description>   # e.g. feat/qwen3-omni
```

If already on a feature branch, confirm it is the one you want to publish.

## 3. Validate locally

Run the narrowest meaningful checks first.

```bash
pre-commit run --all-files
```

Add targeted tests or smoke tests when they fit the change better than repo-wide checks.

## 4. Stage

```bash
git add lmdeploy/path/to/changed_file.py     # specific files only, never git add .
git status                                   # verify staged set
```

Stage only the files intended for this PR.

## 5. Commit

```bash
git commit -m "feat: add Qwen3-Omni support"
# Conventional prefixes: feat | fix | refactor | docs | test | chore
```

Before committing, confirm:

- The staged diff matches the intended PR scope.
- Validation results are known.
- No unrelated files are included.

## 6. Push

```bash
git push -u origin <branch>
```

If the branch already exists on the remote, a plain `git push` may be enough.

## 7. Create PR

`<env>` = `fp8` for lmdeploy_fp8, `vl` for lmdeploy_vl.

```bash
conda run -n <env> gh pr create \
  --repo InternLM/lmdeploy \
  --title "<type>: <short description>" \
  --body "$(cat <<'EOF'
## Summary
- <bullet 1>
- <bullet 2>

## Test plan
- [ ] `pre-commit run --all-files` passes
- [ ] unit tests pass: `pytest tests/test_lmdeploy/`
- [ ] manual smoke test with `0_pipe.py` (VLM) or pipeline

Generated with Codex
EOF
)"
```

Before `gh pr create`, confirm the base branch, head branch, and committed contents are correct.

## Output Contract

This skill produces:

- PR URL (printed by `gh pr create`)
- Branch name used
- Commit SHA (`git rev-parse HEAD`)
- Validation status (pre-commit/tests before commit)
