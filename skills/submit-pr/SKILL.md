---
name: submit-pr
description: Use when ready to open a new pull request against InternLM/lmdeploy — covers creating a branch off main, linting, staging specific files, committing with a conventional prefix, pushing, and creating the PR via gh.
---

# Submit a PR for LMDeploy

## 1. Create branch (off main)

Skip this step if already on a feature branch.

```bash
git checkout main && git pull
git checkout -b <type>/<short-description>   # e.g. feat/qwen3-omni
```

## 2. Lint

```bash
pre-commit run --all-files
```

## 3. Stage

```bash
git add lmdeploy/path/to/changed_file.py     # specific files only, never git add .
git status                                   # verify staged set
```

## 4. Commit

```bash
git commit -m "feat: add Qwen3-Omni support"
# Conventional prefixes: feat | fix | refactor | docs | test | chore
```

## 5. Push

```bash
git push -u origin <branch>
```

## 6. Create PR

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Output Contract

This skill produces:

- PR URL (printed by `gh pr create`)
- Branch name used
- Commit SHA (`git rev-parse HEAD`)
- Lint status (pre-commit pass/fail before commit)
