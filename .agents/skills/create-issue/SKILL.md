---
name: create-issue
description: Create a GitHub issue in this repository from its issue forms (Feature / improvement, Bug report), with checkable "Done when" criteria. Use when the user asks to create, raise, open or file an issue, report a bug, or turn an idea, note, Miro sticky or conversation into a ticket. Not for pull requests or for editing an existing issue.
---

# Create an issue

Create an issue that looks exactly like one filed through the repository's issue forms in the browser. The forms in `.github/ISSUE_TEMPLATE/` are the source of truth: read them every time, never rely on a remembered list of fields.

Run every command from the repository root. You need `gh` (authenticated) and `uv`.

## 1. Read the forms

```bash
uv run .agents/skills/create-issue/scripts/render_issue.py list
```

This prints each template's name, title prefix, labels and fields (label, type, required, description, default value, options).

Pick the template that fits: a **Bug report** for something that isn't working, a **Feature / improvement** for new work. If the request fits neither, or you can't tell, ask the user.

## 2. Search for duplicates

```bash
gh issue list --state all --search "<two or three key words>" --limit 10
```

If a close match exists, show it to the user and ask whether to comment on it instead. Only tick the form's "I have searched the existing issues" box after you have actually searched.

## 3. Draft the answers

Fill each field from what the user said and from the conversation. Follow each field's description from step 1.

- **Title**: short and specific, without the template prefix (the script adds it).
- **Context / bug fields**: the user's facts, not your guesses. Link the source (Miro sticky, decisions doc, meeting notes, failing run) when there is one.
- **Done when**: keep the template's default structure (**Must have** / **Nice to have**). Propose concrete criteria drawn from the conversation, each one checkable by someone else without asking. Never present invented criteria as agreed: the user confirms or edits them in step 4.
- Leave optional fields empty rather than padding them; they render as `_No response_`, as on GitHub.
- Never put credentials, tokens or personal data in an issue.

Write the answers to a temporary file outside the repository, keyed by field label, with checkbox fields as a list of the options to tick:

```json
{
  "title": "Short summary",
  "Is there an existing issue for this?": ["I have searched the existing issues"],
  "Context": "...",
  "Done when": "**Must have**\n\n- [ ] ...\n\n**Nice to have**\n\n- [ ] ..."
}
```

Render it:

```bash
uv run .agents/skills/create-issue/scripts/render_issue.py render <template> <answers.json> <tmp-dir>
```

The script checks required fields and unknown labels, writes `body.md`, and prints the final `title`, `labels` and `body_file`. Fix and re-render on any error.

## 4. Confirm with the user

Creating an issue is visible to the whole team. Show the user the final title, labels and the full rendered body, and ask them to confirm. Apply any edits and re-render. Do not create the issue without a clear yes.

## 5. Create it

```bash
gh issue create --title "<title>" --body-file <body_file> --label "<label>" [--label "<label>" ...]
```

Pass each label from the render output with its own `--label`. Report the issue URL to the user.

Then offer, without doing it unasked, to create the linked branch so the work can start (`gh issue develop <number> --checkout`).
