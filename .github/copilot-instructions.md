Code review instructions:

- All code (names, comments, docstrings) must be written in English. Proofread for typos.
- Code must be clean, idiomatic Python (PEP 8) and consistent with the existing codebase.
- Names must be descriptive: verbs for functions (`split_channels`), nouns for values (`channel_index`), predicates for booleans (`is_grayscale`, `has_alpha`).
- Functions should be small and single-purpose, with nesting of 2 (max 3) levels and at most 3 arguments (4 if strictly needed).
- Comments only explain non-obvious decisions or trade-offs; never restate the code.
- Architecture rules:
    - Dependency direction is `ui → tools → core`. `forensics_app/core` must never import `tkinter`.
    - Tools subclass `ForensicsTool`, return a `ToolResult` (or `None` on cancel), and must never call `document.apply()` or mutate `document.current`.
    - Numerical work lives in plain functions (PIL/NumPy in, NumPy out) and must have unit tests in `tests/`.
    - New tools must be registered in `build_tool_registry()`.
    - Images can arrive in any PIL mode (`RGB`, `RGBA`, `L`, `P`...); conversions must be explicit and unsupported modes must raise a clear error.
    - Saving must never overwrite the original evidence image.
- New dependencies must be added to both `requirements.txt` and `pyproject.toml`.
- Prefer clarity, correctness and maintainability over cleverness.
- In review comments, use `inline code formatting` for code elements.

