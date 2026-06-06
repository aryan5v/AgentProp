# Local-Only Notes (not committed)

Copy this file to `docs/local/README.md` (or any path under `docs/local/`) for
your own working notes. The entire `docs/local/` directory is gitignored.

Keep **internal** material here — anything that reads like notes to the team,
not documentation for users or contributors:

- paper outlines and research backlogs
- release/publishing runbooks and PyPI token setup
- benchmark launch bundles, runbooks, and preflight manifests
- regression postmortems and "what we should do next" roadmaps
- extended case-study analysis companions (beyond public `REPORT.md` summaries)
- `agentprop readiness --out docs/local/readiness.md` rollout checklists

## Suggested layout

```text
docs/local/
  README.md                 # your index
  publishing.md             # PyPI / release checklist
  paper-outline.md          # draft paper structure
  research-backlog.md       # future work
  terminal-bench/           # launch runbooks, extra-instructions snapshots
  case-studies/             # extended findings beyond public reports
```

## Public repo rule

If it contains future work, blockers, "remaining", or maintainer-only ops, it
belongs under `docs/local/`, not under `docs/` in git.
