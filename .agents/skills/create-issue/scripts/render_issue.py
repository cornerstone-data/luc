# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""Render a GitHub issue from the repository's issue forms.

`gh issue create` cannot fill an issue form, so this rebuilds what GitHub produces when a form is
submitted in the browser: the form's title prefix, one `### <label>` section per field, and the
form's labels. Fields are read from `.github/ISSUE_TEMPLATE/` at runtime, so template edits need
no change here.

    uv run render_issue.py list
    uv run render_issue.py render <template> <answers.json> <out-dir>

`answers.json` maps field labels to values, plus a `title` key:

    {"title": "Short summary", "Context": "...", "Done when": "**Must have**\\n- [ ] ..."}

Checkbox fields take a list of the option labels to tick. `render` writes `<out-dir>/body.md` and
prints JSON with `title`, `labels` and `body_file` for `gh issue create`.
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

NO_RESPONSE = "_No response_"


def template_dir() -> Path:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(root) / ".github" / "ISSUE_TEMPLATE"


def load_forms() -> dict[str, dict]:
    forms = {}
    for path in sorted(template_dir().glob("*.y*ml")):
        form = yaml.safe_load(path.read_text())
        if isinstance(form, dict) and "body" in form:
            forms[path.stem] = form
    return forms


def input_fields(form: dict) -> list[dict]:
    """The fields that appear in the issue body; markdown blocks are display-only."""
    return [item for item in form["body"] if item["type"] != "markdown"]


def describe(forms: dict[str, dict]) -> list[dict]:
    out = []
    for key, form in forms.items():
        fields = []
        for item in input_fields(form):
            attrs = item.get("attributes", {})
            field = {
                "label": attrs["label"],
                "type": item["type"],
                "required": item.get("validations", {}).get("required", False),
                "description": attrs.get("description", "").strip(),
            }
            if "value" in attrs:
                field["default"] = attrs["value"]
            if item["type"] in ("checkboxes", "dropdown"):
                field["options"] = [
                    o["label"] if isinstance(o, dict) else o for o in attrs["options"]
                ]
                if item["type"] == "checkboxes":
                    field["required"] = any(
                        isinstance(o, dict) and o.get("required")
                        for o in attrs["options"]
                    )
            fields.append(field)
        out.append(
            {
                "template": key,
                "name": form["name"],
                "description": form.get("description", ""),
                "title": form.get("title", ""),
                "labels": form.get("labels", []),
                "fields": fields,
            }
        )
    return out


def render(form: dict, answers: dict) -> tuple[str, str, list[str]]:
    title = answers.get("title", "").strip()
    if not title:
        sys.exit("answers.json needs a non-empty 'title'")
    prefix = form.get("title", "")
    title = (
        prefix.replace("<title>", title) if "<title>" in prefix else f"{prefix}{title}"
    )

    sections, missing = [], []
    for item in input_fields(form):
        attrs = item["attributes"]
        label = attrs["label"]
        value = answers.get(label)
        if item["type"] == "checkboxes":
            ticked = set(value or [])
            options = attrs["options"]
            if any(o.get("required") and o["label"] not in ticked for o in options):
                missing.append(label)
            text = "\n".join(
                f"- [{'X' if o['label'] in ticked else ' '}] {o['label']}"
                for o in options
            )
        else:
            text = (value or "").strip()
            if not text and item.get("validations", {}).get("required"):
                missing.append(label)
            text = text or NO_RESPONSE
        sections.append(f"### {label}\n\n{text}")

    unknown = (
        set(answers)
        - {"title"}
        - {i["attributes"]["label"] for i in input_fields(form)}
    )
    if unknown:
        sys.exit(f"unknown fields in answers.json: {sorted(unknown)}")
    if missing:
        sys.exit(f"required fields missing: {missing}")
    return title, "\n\n".join(sections) + "\n", form.get("labels", [])


def main() -> None:
    forms = load_forms()
    if len(sys.argv) == 2 and sys.argv[1] == "list":
        print(json.dumps(describe(forms), indent=2, ensure_ascii=False))
    elif len(sys.argv) == 5 and sys.argv[1] == "render":
        key, answers_path, out_dir = sys.argv[2:]
        if key not in forms:
            sys.exit(f"unknown template {key!r}; choose from {sorted(forms)}")
        title, body, labels = render(
            forms[key], json.loads(Path(answers_path).read_text())
        )
        body_file = Path(out_dir) / "body.md"
        body_file.parent.mkdir(parents=True, exist_ok=True)
        body_file.write_text(body)
        print(
            json.dumps(
                {"title": title, "labels": labels, "body_file": str(body_file)},
                ensure_ascii=False,
            )
        )
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
