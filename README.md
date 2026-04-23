# LMDeploy-Skills

Personal Claude Code skills for LMDeploy development.

## Skills

### `/check-env`
Verify editable install, Python path, and CUDA. Run at session start or when `import lmdeploy` fails.

### `/code-navigation`
Full LMDeploy directory map. Covers PyTorch backend layout and key files. Read this first when navigating an unfamiliar part of the codebase.

### `/support-new-model`
Step-by-step guide for adding a new LLM or VLM to the PyTorch backend: model file → module_map → config builder → VL preprocessor → archs.py. Includes code skeletons, checklist, common pitfalls, and verification commands.

### `/submit-pr`
Full PR workflow: branch off main, lint, stage specific files, commit (conventional prefix), push, `gh pr create` with summary + test plan. Targets `InternLM/lmdeploy`.

### `/resolve-review`
Fetch PR inline and top-level review comments via `gh api`, fix each, lint, commit, push.

### `/karpathy-guidelines`
Coding behavior: think before coding, minimum code, surgical edits only, define verifiable success criteria. Apply at the start of any non-trivial implementation task.

---

## Wiring to a repo

Add to `.claude/settings.json` in the repo:

```json
{
  "skillsDirectories": ["/nvme1/zhouxinyu/LMDeploy-Skills/skills"]
}
```
