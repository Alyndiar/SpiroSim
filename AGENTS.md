# AGENTS.md – How to work on this repository (SpiroSim)

> 📝 **Note pour les humains (FR)**
>
> Ce dépôt contient le simulateur de Spirograph **SpiroSim**.  
> Nous utilisons Codex / ChatGPT pour nous aider à écrire et modifier le code.
> - L’IA ne doit **jamais** travailler directement sur `main`, seulement sur une branche `feature/*`.
> - Les changements doivent suivre la demande précise.
> - Les commentaires doivent être **gardés à jour**.
> - Pour les refactors importants, il faut s’assurer que toutes les fonctions et modules affectés sont bien mis à jour.
>
> Le reste de ce fichier est écrit pour les agents (Codex / ChatGPT) en anglais.

---

## Project overview

- This repository contains **SpiroSim**, a Spirograph simulation / drawing application.
- Main language: **Python**.
- The project focuses on:
  - Simulating Spirograph wheels, rings and modular tracks.
  - Providing a GUI for configuring layers, tracks and colors.
  - Supporting modular Super Spirograph–style pieces.

When you make changes, **preserve the existing behaviour** unless the user explicitly asks for a change.

---

## How to run the project (SpiroSim-specific)

There is currently **no automated test suite**. All validation is done via **manual testing** in the GUI.

Typical way to run the application locally:

- From the repository root:
  - `python SpiroSim.py`  
  (or any other entrypoint mentioned in the README, if it differs).

When you modify the code, you should propose simple **manual checks**, for example:

- Start the application.
- Open the relevant menu (e.g. layer editor, modular track editor).
- Change the parameters you affected (e.g. radial spacing, tooth spacing, modular notation).
- Verify that:
  - The UI updates as expected.
  - The drawing behaves consistently with the parameters.
  - No errors are raised in the console.

Do **not** invent automated test commands or CI workflows unless the user requests them.

---

## Coding conventions

- Follow the existing style and structure in this repository.
- **Keep comments up to date**:
  - If you change logic, update or remove any outdated comments.
  - Add short clarifying comments where behaviour is non-obvious.
- Prefer small, incremental changes instead of large, sweeping rewrites.
- Avoid reformatting entire files unless explicitly requested.

If you introduce new functions or classes:

- Use clear, descriptive names.
- Add short docstrings or comments describing the purpose and parameters when helpful.

---

## Branching and Git workflow

- **Never work directly on `main`.**
- Assume changes are developed on a **feature branch**:
  - Use a branch name like: `feature/<short-description>`  
    (e.g. `feature/modular-track-editor-button`).

When describing Git steps, prefer:

1. Creating or checking out a `feature/*` branch.
2. Making changes and committing them in small, logical steps.
3. Opening or updating a Pull Request from the feature branch into `main`.

Do **not** suggest force pushes or rewriting history on `main`.

---

## Change management and planning

Before making non-trivial changes, you should:

1. **Create a short plan**:
   - List the steps you intend to take.
   - Identify which files and functions will be touched.
2. Then implement the plan **step by step**, updating it if needed.

When a **major refactor** is requested:

- Identify all affected modules, classes and functions.
- Ensure changes are **fully propagated**:
  - Update all callers and related code paths.
  - Keep behaviour consistent unless the user asks for a change.
- Clearly explain:
  - What was refactored.
  - Which parts of the codebase were updated.
  - Any follow-up manual tests that should be run.

---

## Areas to be careful with

- Core drawing logic (gears, rings, modular track logic).
- UI wiring between:
  - Layer configuration,
  - Track configuration,
  - Modular track notation and modular track editor.

When modifying these areas:

- Keep changes minimal and well-motivated.
- Preserve existing parameter semantics unless the user explicitly wants them changed.
- Update any related comments so they remain accurate.

---

## How to help best in this repo

When the user asks for a feature or fix:

1. Restate the request in your own words.
2. Propose a **short implementation plan**.
3. Apply the changes in small steps, explaining each step.
4. Keep comments accurate and up to date.
5. Provide a concise list of manual checks the user can run to validate the change.

Always adapt the scope of your changes to **exactly** what the user requested, unless they explicitly invite broader refactoring.
