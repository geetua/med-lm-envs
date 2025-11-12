# MedicationQA

### Overview
- **Environment ID**: `medicationqa`
- **Short description**: Single-turn medication question answering from the MedInfo 2019 Medication QA dataset. Evaluates how well a model answers consumer-style medication questions.

### Dataset
- **Source**: https://github.com/abachaa/Medication_QA_MedInfo2019
- **Splits**: the source has no predefined splits; this environment currently exposes the full dataset as `test`.
- **Splits**: The original dataset does not include predefined train/validation/test partitions.
  This environment currently exposes the full dataset as the `test` split for evaluation consistency.
  (Future versions may introduce explicit splits if needed.)
- **Split sizes**:
  - Test: 690  _(full dataset; no official train/val)_
  
### Task
- **Type**: Single-Turn
- **Rubric**: LLM-as-a-Judge (adapted from MedHELM / MedDialog)
- **Evaluation dimensions**:
  - Accuracy (1–5)
  - Completeness (1–5)
  - Clarity (1–5)

### Quickstart

```bash
uv run vf-eval medicationqa
