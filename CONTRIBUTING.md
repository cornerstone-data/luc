# Contributing to Cornerstone LUC

Thanks for taking the time to contribute. This guide covers how work moves from an idea to a merged change: raising an issue, branching, opening a pull request, and getting it reviewed.

- [Issues first](#issues-first)
- [Workflow](#workflow)
- [Development setup](#development-setup)
- [Commit messages and PR titles](#commit-messages-and-pr-titles)
- [Pull request guidelines](#pull-request-guidelines)
- [Code review](#code-review)

## Issues first

Who needs an issue depends on who you are:

- **External contributors** must raise an issue before opening a pull request, so the scope and the closing criteria are agreed with the core team before code is written. Pull requests without an agreed issue may be closed.
- **Core developers** (the Cornerstone team, with write access to the repository) can open a pull request directly. The PR description then carries the context and the **Done when** criteria that an issue would otherwise hold.

When raising an issue:

- Always use one of the [issue templates](https://github.com/cornerstone-data/luc/issues/new/choose): **Feature / improvement** for new work, **Bug report** for something that isn't working. Blank issues are disabled.
- Search the existing issues first to avoid duplicates.
- Fill in **Done when** with checkable criteria:
  - **Must have**: everything that has to be true before the issue can be closed.
  - **Nice to have**: welcome but not blocking; move leftovers to a follow-up issue rather than keeping the issue open.
- If no template fits, open a Feature / improvement issue asking for a new template.

## Workflow

We use [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow): all changes go through pull requests.

1. **Create an issue** using a template (core developers may skip this and steps 2 and 7, and branch directly).
2. **Create a branch from the issue** (*Development → Create a branch* in the issue sidebar), so the branch name carries the issue number, e.g. `52-rescale-gpw-livestock`.
3. **Commit and push** to that branch. Several small commits are fine; they are squashed on merge.
4. **Open a pull request** into `main` and, if there is an issue, link it with `Closes #<issue>` in the description.
5. **Get a review** and address the comments (see [Code review](#code-review)).
6. **Squash and merge.** Replace GitHub's default merge message with the PR title, so `main` keeps one readable commit per change.
7. **The issue closes automatically** through the `Closes #<issue>` link, and the branch is deleted.

`main` is protected: no direct pushes or force-pushes, and history stays linear (squash or rebase merges only).

## Development setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). See the [README](README.md#getting-set-up) for storage configuration (`INGEST_ROOT`, `SCRATCH_ROOT`).

```bash
uv sync    # install dependencies, including the dev group
```

### CI

CI runs the checks on every pull request: linting and formatting, import layers, type checking and unit tests. You don't need to run them locally; open the PR and check the CI result.

- If CI fails, fix the cause and push again.
- Never skip or disable a check or test to make CI pass.

Things CI enforces that are easy to trip on:

- **Import layers.** The layer contracts in `pyproject.toml` are checked; in particular the pipeline (`jdluc`) must never import the validator (`validation`).
- **Lock file.** If you change dependencies in `pyproject.toml`, commit the updated `uv.lock`.
- **Formatting.** Python is checked with `ruff format` and Markdown with `mdformat` (no line wrapping); unformatted files fail CI.
- **Large files.** Data and outputs do not belong in git; keep them under your `INGEST_ROOT` / `SCRATCH_ROOT`.

Integration tests need live GCS credentials and are not run by CI; run them explicitly with `uv run pytest -m integration` when your change touches cloud storage.

## Commit messages and PR titles

PR titles become the commit messages on `main`, so write them as [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <what changes, in the imperative>
```

| Type       | Use for                                          |
| ---------- | ------------------------------------------------ |
| `feat`     | new capability in the pipeline or tooling        |
| `fix`      | a bug fix                                        |
| `docs`     | documentation only                               |
| `refactor` | code change with no behaviour change             |
| `test`     | adding or fixing tests                           |
| `chore`    | maintenance: dependencies, CI, release touch-ups |

Examples from the history: `feat: ingest gpw livestock dataset`, `fix: include Taiwan in outputs`, `docs: update tooling docstring and further_research`.

## Pull request guidelines

A pull request should let a reviewer understand the change without asking. At minimum, the description covers:

- **Problem**: what the change solves, with `Closes #<issue>` when there is one.
- **Change**: what was done and why this approach.
- **Done when**: the must-haves (from the issue, or written here when there is none), ticked off as they are met.
- **Testing**: tests added or updated, and any pipeline run used to check the result (countries / tiles, methodology).
- **Impact on outputs**: if emissions factors or other published outputs change, say which and by roughly how much.
- **Dependencies**: other PRs or issues this one depends on.

Also:

- Keep pull requests focused on one issue. Split unrelated changes into their own issue and PR.
- New features come with tests; bug fixes come with a regression test where possible.
- Update the docs in `docs/` when methodology, data sources or behaviour change.
- Open a draft PR early if you want feedback before the work is finished.

## Code review

A PR can merge into `main` once it has one approving review, every review thread is resolved, CI passes, and the branch is up to date with `main`.

This guide does not cover AI-assisted contributions yet; guidelines for AI coding agents will be agreed separately.

`needs-rebase` is set and cleared automatically when a PR conflicts with or falls behind `main`; rebase on `main` when you see it.

When reviewing and addressing comments:

- Ask for review from someone who knows the domain (methodology, datasets) and, when useful, someone who knows the code area well.
- Review every comment before asking for another round, and check whether the same issue appears elsewhere in the PR.
- If you won't apply a suggestion, reply with the reason. There is no need to reply "done" on each comment; reviewers assume suggestions are applied unless you say otherwise.
- Postponing a change is fine: open a follow-up issue and link it from the comment.
- When you are ready for another round, do a self-review and leave a comment to tell the reviewer.

Thank you to everyone who has contributed to Cornerstone LUC.
