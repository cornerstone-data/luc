# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Copilot, …) working in this repository. They apply on top of [CONTRIBUTING.md](CONTRIBUTING.md), which humans follow too; where the two overlap, they say the same thing.

## Who you are working for

You act on behalf of the person who started the session, and you inherit their role:

- **Core developer** (Cornerstone team, write access): you may open a pull request without an issue. Put the context and the **Done when** criteria in the PR description instead.
- **External contributor**: an issue agreed with the core team must exist before you open a pull request. If there is none, stop and draft the issue with the user instead of the PR.

When you don't know which applies, ask.

## Working principles

- **Think first.** State your assumptions. If the request can be read more than one way, surface the readings instead of picking one silently. Push back when a simpler approach exists.
- **Keep it simple.** Write the minimum code that solves the problem: no speculative features, abstractions, or error handling for cases that can't happen.
- **Change only what the task needs.** Match the existing style. Remove only what your own change made unused; flag pre-existing dead code instead of deleting it.
- **Verify.** Turn the task into checkable goals: write the failing test, then make it pass; make sure tests pass before and after a refactor.

## Issues

To create an issue, follow the `create-issue` skill: [.agents/skills/create-issue/SKILL.md](.agents/skills/create-issue/SKILL.md). Read it even if your tool doesn't load skills automatically.

- Create issues through the templates in `.github/ISSUE_TEMPLATE/` (**Feature / improvement** or **Bug report**); blank issues are disabled.
- Search existing issues before creating one.
- **Done when** must contain checkable criteria under **Must have** and, optionally, **Nice to have**. Write each item so a reviewer can verify it without asking.

## Branches and commits

- Never commit or push to `main` directly. Work on a branch: `<issue-number>-<short-slug>` when there is an issue, `<short-slug>` otherwise.
- Commit or push only when the user asks.
- Keep each branch and PR to one issue or one change. Raise unrelated findings as a separate issue or tell the user; don't fold them in.
- Never commit data, model outputs, credentials or `.env`. Data lives under `INGEST_ROOT` / `SCRATCH_ROOT`.
- If you change dependencies in `pyproject.toml`, commit the updated `uv.lock`.

## CI

CI runs the lint, type and test checks on every pull request. Rely on it rather than running them locally:

- After opening or updating a pull request, check the CI result.
- If CI fails, fix the cause and push again. If you can't fix it, say so in the PR.
- Never skip or disable a check, test or type check to make CI pass.

## Pull requests

**Title**: [Conventional Commits](https://www.conventionalcommits.org/), imperative, because it becomes the squash-merge commit on `main`:

```
<type>: <what changes>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

**Description**: write it yourself, covering these sections in this order:

1. **Problem**: what the change solves, with `Closes #<issue>` when there is an issue.
2. **Change**: what was done and why this approach.
3. **Done when**: the must-haves (from the issue, or written here when there is none), ticked off only when actually met; nice-to-haves left open or moved to a follow-up issue.
4. **Testing**: tests added or updated, checks run and their outcome, and any pipeline run used to verify (countries / tiles, methodology).
5. **Impact on outputs**: whether emissions factors or other published outputs change, which ones, and by roughly how much; "None" if not.
6. **Dependencies**: other PRs or issues this one depends on; "None" if not.

Also:

- State in the description that the PR was prepared with an AI agent, and which one.
- Open it as a **draft** unless the user asks for a ready-for-review PR.
- Don't merge, approve, or enable auto-merge; a human reviews and merges.
- Update `docs/` when methodology, data sources or behaviour change.

## Code review

Review happens in two stages, tracked with labels:

1. **AI review.** An agent reviews the PR and posts one comment ending in a verdict, then labels it:
   - `llm-changes-needed`: problems found; the author or agent iterates, then asks for another AI review.
   - `ready-for-human-review`: no blocking problems; a human reviewer takes over.
2. **Human review.** Merging into `main` needs one approving review, every review thread resolved, CI passing, and the branch up to date with `main`.

`needs-rebase` is set and cleared automatically when a PR conflicts with or falls behind `main`; rebase on `main` when you see it.

When reviewing a PR as an agent, swap the labels instead of adding a second one, and never approve: approval is a human's call.

When addressing review comments:

- Address every comment, or reply with the reason you didn't.
- Don't reply "done" on each comment; push the changes and leave one summary comment.
- Check whether the same problem appears elsewhere in the PR.
- If a fix is postponed, open a follow-up issue and link it in the reply.
