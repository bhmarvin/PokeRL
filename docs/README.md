# Docs

This folder contains short reference documents for the current PokeRL codebase.

## Files

- `observation-tensor.md`: full 758-dim observation layout and feature blocks
- `reward-function.md`: reward shaping design, current config, tactical levers, and design rationale
- `code-reference.md`: main files, symbols, architecture, data flow, and common commands

## Suggested Reading Order

1. `observation-tensor.md`
2. `reward-function.md`
3. `code-reference.md`

## Notes

- These docs reflect the current code including: 758-dim observation vector with force_switch flag, reward rebalance (victory_value=15.0, halved penalties, disabled wasted_free_switch), removal of Welford normalization, and wider encoder/deeper MLP architecture (705-dim extractor output, [512,256,128] MLP).
