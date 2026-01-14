import localisation


def language_entry(code: str) -> tuple[str, str, str]:
    display_name = localisation.language_display_name(code)
    readme_path = localisation.resolve_readme_path(code)
    rel_path = readme_path.relative_to(localisation.LOCALISATION_DIR.parent)
    return code, display_name, rel_path.as_posix()


def main() -> None:
    languages = localisation.available_languages()
    preferred = [code for code in ("en", "fr") if code in languages]
    remaining = sorted(code for code in languages if code not in preferred)
    ordered = preferred + remaining

    lines = ["# Languages", "", "This repository includes the following localizations:", ""]
    for code in ordered:
        code_value, display_name, rel_path = language_entry(code)
        lines.append(f"- [{display_name}]({rel_path}) (`{code_value}`)")

    content = "\n".join(lines) + "\n"
    output_path = localisation.LOCALISATION_DIR.parent / "LANGUAGES.md"
    output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
