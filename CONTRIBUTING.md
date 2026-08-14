# Contributing to muse.cpp

muse.cpp is a privately maintained fork of llama.cpp. Contributions should serve this fork's runtime, hardware, packaging, and deployment requirements.

## AI-assisted contributions

AI-assisted work is allowed for code, tests, documentation, commit messages, issues, pull requests, and review discussions. This fork does not require AI disclosure or a special commit trailer.

Contributors remain responsible for validating changes before they are merged. The repository may use automated agents to implement, test, package, and publish changes when those actions are authorized by the repository owner.

## Workflow

- An issue is optional for features and fixes.
- There is no limit on concurrent pull requests.
- CPU, CUDA, server, and multimodal changes may be developed together when that is the practical scope of the change.
- Keep unrelated changes separate when doing so makes review, testing, or rollback safer.
- Describe the behavior change, relevant tradeoffs, and validation performed.
- Preserve compatibility where practical, but fork-specific requirements may intentionally differ from upstream llama.cpp.

## Testing

- Build the affected targets.
- Run focused regression tests for the changed behavior.
- For `ggml` operator changes, run `test-backend-ops` against the affected backends.
- For performance changes, record the model, quantization, hardware, command-line options, prompt size, concurrency, and before/after measurements.
- For packaging changes, verify runtime dependencies, embedded CUDA architectures, archive checksums, and a clean launch from the package directory.

## Coding guidelines

- Follow the style of surrounding code.
- Prefer simple changes and existing infrastructure.
- Avoid unnecessary dependencies and abstractions.
- Keep comments concise and limited to non-obvious behavior.
- Consider supported operating systems, CPU architectures, CUDA architectures, and other backends as appropriate to the change.
- Use sized integer types in public APIs when the size is part of the interface contract.
- Add tests for new operators, data types, scheduling rules, and bug fixes when practical.

See [AGENTS.md](AGENTS.md) for repository working rules and validation expectations.

## Submitting changes elsewhere

These rules apply only to this fork. A contribution sent to upstream llama.cpp or another repository must follow that target repository's contribution policy.
