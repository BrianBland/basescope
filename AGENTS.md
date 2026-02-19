# AGENTS.md

## Build & Verify

```bash
cargo build          # must exit 0
cargo clippy         # must produce zero warnings
cargo test           # must pass all tests
```

Always run all three before considering any change complete.

## Key Rules

- Never mix fee units within a single axis or histogram group — pick one via `pick_fee_unit()`.
- Color blending is gamma-correct RGB — don't reintroduce HSV.
- No new dependencies without justification.
- Don't refactor while fixing bugs.

## Keeping Docs in Sync

Any changes to project structure, modules, or architecture must be reflected in both this file and `README.md`.
Any changes to usage (keybindings, CLI flags, environment variables, features) must be reflected in `README.md`.
