# Instructions for muse.cpp

This repository is a privately maintained fork of llama.cpp. Upstream llama.cpp contribution restrictions do not apply to work that stays in this fork.

## AI-assisted work

AI-assisted development is allowed for all repository work, including:

- Reading, debugging, designing, and implementing code
- Writing tests and documentation
- Writing commit messages, issue content, pull request descriptions, and review replies
- Creating commits, branches, issues, and pull requests when the user authorizes the corresponding repository action
- Running builds, tests, benchmarks, packaging, and deployment checks

No AI disclosure or special commit trailer is required by this fork. If a change is submitted to another project, follow that target project's rules.

## Scope and repository operations

- Work autonomously within the scope requested by the user.
- Do not infer authorization for external publication, pushing, merging, or destructive operations that the user did not request.
- Preserve unrelated user changes in a dirty worktree.
- Do not use destructive Git commands such as `git reset --hard` or discard user changes unless the user explicitly requests it.
- Inspect relevant code and existing patterns before editing.
- Prefer focused changes that reuse existing infrastructure.

## Code standards

- Keep source code and code comments ASCII unless a file or feature requires other characters.
- Keep comments concise and explain only non-obvious behavior or invariants.
- Do not add comments that merely restate the code.
- Do not hard-wrap prose or split a sentence only to meet a fixed line width.
- Follow the style and structure of surrounding code.
- Avoid unnecessary dependencies, files, abstractions, and invasive subsystems.
- Keep cross-platform and multi-backend behavior in mind unless the change is explicitly backend-specific.
- Use the existing Jinja implementation in `common/jinja`; llama.cpp does not use Minja.

## Validation

- Run tests and builds in proportion to the risk and scope of the change.
- Add or update regression coverage for behavior changes when practical.
- For `ggml` operator changes, use `test-backend-ops` to compare backend results.
- For CUDA changes, validate every supported CUDA architecture that can be checked in the available environment and clearly report any architecture that was build-only.
- Run `git diff --check` before handing off source changes.

## Useful resources

- [Contributing guide](CONTRIBUTING.md)
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md)
- [Model development documentation](docs/development/HOWTO-add-model.md)
- [PEG parser documentation](docs/development/parsing.md)
- [Auto parser documentation](docs/autoparser.md)
- [Jinja documentation](common/jinja/README.md)
