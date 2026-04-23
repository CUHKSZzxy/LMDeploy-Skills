# LMDeploy-Skills

Personal Claude Code skills for LMDeploy development.

## Skills

### `/check-env`

Use when `import lmdeploy` fails, CUDA not found, wrong Python version, editable install not recognized, or `conda run` invokes system Python. Covers conda activation, editable install verification, and CUDA sanity check.

### `/code-navigation`

Use when you need to locate a specific subsystem, module, or key file in the LMDeploy codebase. Covers pytorch backend, vl/, serve/, and top-level orchestration files.

### `/support-new-model`

Use when adding support for a new LLM or VLM architecture to LMDeploy's PyTorch backend. The SKILL.md is a lean workflow with step summaries; deep content lives in `references/` and is loaded only when needed:

| Reference file                    | Load when                                                   |
| --------------------------------- | ----------------------------------------------------------- |
| `references/key-files.md`         | Before writing any code — study guide + file table          |
| `references/llm-code-skeleton.md` | Implementing Step 1 (model file) or Step 3 (config builder) |
| `references/vlm-preprocessor.md`  | Implementing Step 4 (VL preprocessor)                       |
| `references/pitfalls.md`          | Anything fails or produces wrong outputs                    |

### `/submit-pr`

Use when ready to open a new pull request against `InternLM/lmdeploy`. Covers branch creation, lint, staged commit, push, and `gh pr create`. Returns: PR URL, branch name, commit SHA, lint status.

### `/resolve-review`

Use when a PR has inline or top-level review comments to address. Fetches all comments via `gh api`, guides through fixing each, then lints, commits, and pushes. Returns: list of comments addressed, commit SHA, push confirmation.

### `/karpathy-guidelines`

Use at the start of any non-trivial implementation task to set behavioral ground rules: think before coding, minimum code, surgical edits only, define verifiable success criteria.

______________________________________________________________________

## Wiring to a repo

Add to `.claude/settings.json` in the repo:

```json
{
  "skillsDirectories": ["/nvme1/zhouxinyu/LMDeploy-Skills/skills"]
}
```
