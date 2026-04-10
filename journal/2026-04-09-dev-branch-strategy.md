# 2026-04-09 — Switch to `dev` Branch for Staging Deployments

## Background

The `web-ui-implementation` branch previously served as the staging branch — pushes to it
auto-deployed to `dev.sundayalbum.com`. That branch is no longer in active use following the
merge of PR #32. Going forward, a persistent `dev` branch takes its place.

A `dev` branch already exists on the remote (`origin/dev`) but is significantly behind
`main` — its HEAD is an early App Runner infra commit from Phase 3.5, predating all of the
web UI, Lambda pipeline, and color restore work.

## Goal

```
feature/xyz  →  PR to dev  →  auto-deploy to dev.sundayalbum.com
                                        ↓ (when verified)
                              PR to main  →  auto-deploy to prod (app.sundayalbum.com)
```

No manual GitHub Actions clicks for routine dev deploys. `dev` is always a superset of
`main` (reset to `main` after each prod release).

---

## Implementation Plan

### Step 1 — Reset `dev` to `main`

The existing `dev` branch is stale. Hard-reset it to `main` so both environments start from
the same code (including the color restore work from PR #34 and the CI fix from PR #35).

```bash
git checkout dev
git reset --hard origin/main
git push --force-with-lease origin dev
```

`--force-with-lease` is safe here: we own the branch and intentionally want to discard its
old history.

### Step 2 — Update `deploy-web.yml`

Replace the `web-ui-implementation` branch reference with `dev`:

```yaml
# Before
on:
  push:
    branches:
      - main
      - web-ui-implementation

# After
on:
  push:
    branches:
      - main
      - dev
```

Also update the inline comment on the build-and-deploy job (line 30):

```yaml
# Before
# web-ui-implementation → dev App Runner | main → prod App Runner

# After
# dev → dev App Runner | main → prod App Runner
```

The stage-resolution logic (`ref_name == 'main'` → prod, anything else → dev) already works
correctly and needs no change.

### Step 3 — Update `deploy-lambda.yml`

Replace the `web-ui-implementation` branch reference with `dev`:

```yaml
# Before
on:
  push:
    branches: [main, web-ui-implementation]

# After
on:
  push:
    branches: [main, dev]
```

### Step 4 — Verify the workflow triggers

After merging to `main`, make a trivial change on `dev` (e.g. a whitespace edit to any
file covered by the workflow path filters) and push to `origin/dev`. Confirm that:
- `deploy-web.yml` triggers and deploys to `sundayalbum-web-dev` (dev.sundayalbum.com)
- `deploy-lambda.yml` triggers and deploys to the `-dev` suffixed Lambda functions

### Step 5 — Update working convention

Going forward:
- All feature branches are cut from `main`
- Feature PRs target `dev` first (deploys to dev automatically on merge)
- When dev is verified, open a PR from `dev` → `main` (deploys to prod on merge)
- After a prod release, reset `dev` to `main` (Step 1 above, repeated each cycle)

---

## Branch Policy

All changes to the workflow files happen in a feature branch off `main`, merged via PR.
No direct commits to `main` (branch protection is enforced). The `dev` reset in Step 1
is a force-push to `dev` (not `main`) and is safe.

---

## Files to Modify

| File | Change |
|---|---|
| `.github/workflows/deploy-web.yml` | Replace `web-ui-implementation` with `dev` on lines 7 and 30 |
| `.github/workflows/deploy-lambda.yml` | Replace `web-ui-implementation` with `dev` on line 5 |

## What We Are NOT Changing

- Stage-resolution logic in both workflows — `main` → prod, else → dev already works
- App Runner service names, ECR repos, Lambda function names
- Pre-commit hook, test workflows, or any other CI
- `web-ui-implementation` branch — leave it as-is (historical reference)
