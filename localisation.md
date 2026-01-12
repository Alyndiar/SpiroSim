# Localisation system

SpiroSim loads UI translations from the `localisation/` directory. Each language lives in
`localisation/<code>/strings.json`, and the UI chooses a language based on the code selected
in **Options → Language**. If a language is incomplete, missing strings fall back to the base
language and then to English.

## File structure

Each `strings.json` file contains three sections:

- `strings`: UI text labels (menus, dialogs, warnings, etc.).
- `gear_type_labels`: labels used in gear type pickers (`anneau`, `roue`, `triangle`, etc.).
- `relation_labels`: labels for gear relations (`stationnaire`, `dedans`, `dehors`).

Example skeleton:

```json
{
  "strings": {
    "language_name": "English",
    "menu_file": "File"
  },
  "gear_type_labels": {
    "anneau": "ring"
  },
  "relation_labels": {
    "stationnaire": "stationary"
  }
}
```

`language_name` is required, because it is displayed in the language menu.

## Adding a new language

1. Create a new directory under `localisation/` using the language code (for example `es`).
2. Copy `localisation/en/strings.json` and translate it, keeping the same keys.
3. Ensure `strings.language_name` matches the language’s native display name.
4. (Optional) Add a translated README at `localisation/<code>/README.md` (or
   `README.<code>.md` at the repository root). The in-app **Help → Manual** action opens the
   localized README when it exists.
5. Update `LANGUAGES.md` by running:

   ```bash
   python build_languages.py
   ```

6. Commit the new localisation files.

## Language resolution behavior

The localisation loader tries the exact language code first, then the base language, and
finally English. For example, `es_mx` resolves in this order: `es_mx` → `es` → `en`.

