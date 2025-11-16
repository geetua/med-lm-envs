# environments/medicationqa/medicationqa.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests
import verifiers as vf
from datasets import Dataset, load_from_disk
from judge_prompts import JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE
from medarc_verifiers.parsers import JSONParser
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State

# 1. where the data lives upstream
DATA_URL = "https://raw.githubusercontent.com/abachaa/Medication_QA_MedInfo2019/master/MedInfo2019-QA-Medications.xlsx"

# 2. where we’ll cache locally
CACHE_DIR = Path.home() / ".cache" / "medicationqa"
XLSX_PATH = CACHE_DIR / "Medication_QA.xlsx"
DATASET_PATH = CACHE_DIR / "Medication_QA.arrow"

# Scored dimensions must match the keys emitted by judge_prompts.JUDGE_TEMPLATE
JUDGE_DIMENSIONS = ["accuracy", "completeness", "clarity"]


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_xlsx() -> None:
    """Parse the MedicationQA Excel file, normalize columns, and persist it as a Hugging Face
    dataset under the cache directory.

    If the dataset is already present on disk, this function is a no-op.
    """
    if XLSX_PATH.exists():
        return
    resp = requests.get(DATA_URL)
    resp.raise_for_status()
    XLSX_PATH.write_bytes(resp.content)


def build_and_save_dataset() -> None:
    """
    Read the Excel once, normalize columns, and save as a Hugging Face dataset
    under CACHE_DIR. If it's already saved, do nothing.
    """
    # HF datasets saved via save_to_disk write a dataset_info.json in the dir
    dataset_info = CACHE_DIR / "dataset_info.json"
    if dataset_info.exists():
        return

    df = pd.read_excel(XLSX_PATH)

    # normalize column names
    df = df.rename(
        columns={
            "Question": "question",
            "Answer": "answer",
            "Question Type": "question_type",
            "Focus (Drug)": "question_focus",
            "Section Title": "answer_section_title",
            "URL": "answer_url",
        }
    )

    # drop bad rows
    df = df.dropna(subset=["question"]).reset_index(drop=True)

    # add ids
    df["id"] = [f"medicationqa_{i}" for i in range(len(df))]

    # turn into dataset
    ds = Dataset.from_pandas(df)

    # save it
    ds.save_to_disk(str(CACHE_DIR))


def load_dataset_cached() -> Dataset:
    """Load the cached MedicationQA dataset from disk.

    Assumes `build_and_save_dataset()` has already created the dataset in the cache directory.
    """
    ds = load_from_disk(str(CACHE_DIR))
    return cast(Dataset, ds)


def get_split(split: str = "test") -> Dataset:
    """Return the MedicationQA evaluation split.

    The source MedicationQA (MedInfo 2019) release does not provide official
    train/validation/test partitions. For consistency with other single-turn
    environments (e.g. `med_dialog`), this environment currently exposes the
    entire dataset as the `"test"` split regardless of the `split` argument.

    Args:
        split: Requested split name. Accepted for API compatibility but ignored.

    Returns:
        A Hugging Face `Dataset` containing all examples.
    """
    return load_dataset_cached()


# ---------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------


def build_prompt(example: dict) -> str:
    """Construct a model-ready prompt for a single MedicationQA example.

    The prompt instructs the model to answer a consumer medication question
    clearly and safely.
    """
    return (
        "You are a helpful, safety-conscious medical assistant. "
        "Answer the patient's medication question clearly and accurately.\n\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )


def build_metadata(ex: dict) -> dict:
    """Return all non-core fields from a MedicationQA example as metadata."""
    core = {"id", "question", "answer"}
    return {k: v for k, v in ex.items() if k not in core}


def get_eval_example(idx: int = 0) -> dict:
    """Convenience helper to fetch a single example in the format used by evaluators.

    Returns a dict containing the example id, question text, constructed prompt,
    reference answer, and any additional metadata.
    """
    data = get_split("test")
    ex = data[idx]
    return {
        "id": ex["id"],
        "question": ex["question"],
        "prompt": build_prompt(ex),
        "reference_answer": ex["answer"],
        "metadata": build_metadata(ex),
    }


# ---------------------------------------------------------------------
# UTILS FOR JUDGE (same shape as med_dialog)
# ---------------------------------------------------------------------


def _extract_completion_text(completion: Messages) -> str:
    """Extract the assistant’s text content from a chat-style completion."""
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)


def _coerce_score(value: Any) -> float | None:
    """Best-effort conversion of a score value to a float in Python, or `None` if not possible."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _compute_normalized_reward(scores: dict[str, dict[str, Any]]) -> float:
    """Normalize per-dimension judge scores to a single value in [0.0, 1.0].

    Each dimension is expected to be on a 1–5 scale. Scores are clamped to
    [0, 5], divided by 5 to map to [0, 1], and then averaged across the
    dimensions listed in `JUDGE_DIMENSIONS`.
    """

    total_dims = len(JUDGE_DIMENSIONS)
    if total_dims == 0:
        return 0.0

    accumulated = 0.0
    for dimension in JUDGE_DIMENSIONS:
        score = _coerce_score(scores.get(dimension, {}).get("score"))
        if score is None:
            continue
        clamped = max(0.0, min(5.0, score))
        accumulated += clamped / 5.0

    return max(0.0, min(1.0, accumulated / total_dims))


# ---------------------------------------------------------------------
# ENVIRONMENT LOADER (MedDialog-style, MedHELM prompt)
# ---------------------------------------------------------------------


def load_environment(
    cache_dir: Path | str | None = None,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs: Any,
) -> vf.SingleTurnEnv:
    """Load the MedicationQA (MedInfo 2019) evaluation environment.

    This environment:
    - downloads and caches the MedicationQA Excel source,
    - converts it to a Hugging Face dataset (single split),
    - and evaluates model completions with an LLM-as-a-judge rubric adapted
    from MedHELM / MedDialog (accuracy, completeness, clarity).

    Args:
        cache_dir: Optional override for the cache location. Defaults to `~/.cache/medicationqa`.
        judge_model: Model identifier to use for the judge (e.g. "gpt-4o").
        judge_base_url: Optional base URL for a non-OpenAI-compatible endpoint (e.g. Ollama).
        judge_api_key: API key for the judge model. Falls back to the `JUDGE_API_KEY` env var.
        **kwargs: Additional arguments forwarded to `vf.SingleTurnEnv`.

    Returns:
        A configured `verifiers.SingleTurnEnv` ready to be passed to `vf-eval`.
    """
    ensure_cache_dir()
    download_xlsx()
    build_and_save_dataset()
    eval_dataset = load_dataset_cached()

    api_key = judge_api_key or os.getenv("JUDGE_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key)
    judge_parser = JSONParser(fields=list(JUDGE_DIMENSIONS))

    judge_rubric = vf.JudgeRubric(
        parallelize_scoring=True,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
    )

    async def reward_medicationqa(
        prompt: Messages,
        completion: Messages,
        info: Info,
        state: State,
    ) -> float:
        question = str(info.get("question") or "")
        gold_response = str(info.get("answer") or "")  # keep key name consistent with other envs
        model_answer = _extract_completion_text(completion)

        judge_prompt = JUDGE_TEMPLATE.format(
            question=question,
            response=model_answer,
            gold_response=gold_response,
            output_format=JUDGE_OUTPUT_JSON,
        )

        judge_raw = await judge_rubric.judge(
            [{"role": "user", "content": judge_prompt}],
            model_answer,
            gold_response,
            state,
        )

        parsed = judge_parser.parse(str(judge_raw), strip=True)
        if parsed is None:
            parsed = {dim: {"score": None, "explanation": None, "raw": None} for dim in JUDGE_DIMENSIONS}

        normalized = _compute_normalized_reward(parsed)

        judge_record = {
            "scores": parsed,  # per-dimension score + reason
            "raw_judge": str(judge_raw),
        }
        # store judge feedback directly as a top-level state column
        state["judge_feedback"] = judge_record

        # per-example metadata (will be surfaced in results.info)
        info["judge_feedback"] = judge_record
        return normalized

    judge_rubric.add_reward_func(reward_medicationqa, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=eval_dataset,
        eval_dataset=eval_dataset,
        system_prompt="You are a helpful, safety-conscious medical assistant.",
        rubric=judge_rubric,
        name="medicationqa",
        **kwargs,
    )
