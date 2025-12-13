"""
Med-HALT (Reasoning) Environment

Implements a verifiers environment for the Med-HALT dataset's reasoning hallucination tests.
The dataset includes multiple reasoning test types. This environment currently supports:
- reasoning_FCT (False Confidence Test): Evaluates if models can assess proposed answers
- reasoning_nota (None of the Above Test): Tests if models can identify when none of the options are correct
Note: reasoning_fake is not yet supported in this environment.

Paper: https://arxiv.org/abs/2307.15343
Dataset: https://huggingface.co/datasets/openlifescienceai/Med-HALT
"""

from __future__ import annotations

import ast
import json
from typing import Any

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset
from environments.med_halt.prompts import (
    reasoning_fct_prompt,
    reasoning_fct_shots,
    reasoning_nota_prompt,
    reasoning_nota_shots,
)
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice

# The two supported reasoning test types in Med-HALT
REASONING_TEST_TYPES = ["reasoning_FCT", "reasoning_nota"]


def _parse_options(options_str: str) -> dict[str, str]:
    """
    Parse the options field which is stored as a string representation of a dict.

    Args:
        options_str: String representation of options dict

    Returns:
        Dictionary mapping option indices to option text
    """
    try:
        options_dict = ast.literal_eval(options_str)
        if isinstance(options_dict, dict):
            return options_dict
    except (ValueError, SyntaxError):
        pass
    return {}


def _map_example(
    example: dict[str, Any],
    test_type: str,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = None,
    num_few_shot: int = 2,
) -> dict[str, Any] | None:
    """
    Map a Med-HALT example to verifiers format.

    For FCT (False Confidence Test), we route to a specialized mapper
    that asks the model to judge a student's answer as correct/incorrect.

    For NOTA, we treat it as a normal multiple-choice question.
    """
    question = example.get("question", "").strip()
    if not question:
        return None

    # Parse options from string representation
    options_str = example.get("options", "{}")
    options_dict = _parse_options(options_str)

    if not options_dict:
        return None

    # Build a canonical options_list once, shared by FCT and NOTA
    sorted_items = sorted(
        [(k, v) for k, v in options_dict.items() if k != "correct answer"],
        key=lambda x: int(x[0]) if x[0].isdigit() else x[0],
    )
    options_list = [v for _, v in sorted_items]
    if not options_list:
        return None

    # Common indices
    correct_index = example.get("correct_index")
    student_index = example.get("student_index")

    # ---------- FCT PATH ----------
    # FCT examples have a student_index (student's chosen option).
    # We use this as the source of truth instead of relying on the test_type string.
    if student_index is not None:
        if shuffle_answers:
            raise ValueError("shuffle_answers is not supported for reasoning_FCT")
        return _map_fct_example(
            example=example,
            question=question,
            options_list=options_list,
            correct_index=correct_index,
            student_index=student_index,
            num_few_shot=num_few_shot,
        )

    # ---------- NOTA PATH ----------
    # Everything without a student_index is treated as a classic MCQ (reasoning_nota).
    if correct_index is None or correct_index >= len(options_list):
        return None

    return _map_nota_example(
        example=example,
        question=question,
        options_list=options_list,
        correct_index=correct_index,
        shuffle_answers=shuffle_answers,
        shuffle_seed=shuffle_seed,
        num_few_shot=num_few_shot,
    )


def _map_nota_example(
    example: dict[str, Any],
    question: str,
    options_list: list[str],
    correct_index: int,
    shuffle_answers: bool,
    shuffle_seed: int | None,
    num_few_shot: int,
) -> dict[str, Any] | None:
    """
    Map a Med-HALT NOTA example using the author prompt + few-shot examples.

    Output expected: a single JSON object with keys:
      cop, cop_index, why_correct, why_others_incorrect
      - Few-shot: authors use 2-shot by default; we take the first 2 deterministically.
    """
    options_for_prompt: dict[str, Any] = {str(i): opt for i, opt in enumerate(options_list)}
    answer_choice = str(correct_index)

    if shuffle_answers and answer_choice in options_for_prompt:
        options_for_prompt, answer_choice, _ = randomize_multiple_choice(
            options=options_for_prompt,
            answer_choice=answer_choice,
            seed=shuffle_seed,
            row_id=question,  # or example.get("question")
        )
        # update correct_index to match shuffled keys
        correct_index = int(answer_choice)

    def _format_options_kv(opts: dict[str, Any]) -> str:
        return "\n".join(f"{k}: {v}" for k, v in opts.items())

    def _format_shot(shot: dict[str, Any]) -> str:
        inp = shot["input"]
        out = shot["Output"]
        out_json = json.dumps(out, ensure_ascii=False, indent=2)
        return (
            "### Input\n"
            f"Question: {inp['Question']}\n"
            "Options:\n"
            f"{_format_options_kv(inp['Options'])}\n"
            "### Output\n"
            f"{out_json}\n"
        )

    few_shots = reasoning_nota_shots[:num_few_shot]
    few_shot_block = ""
    if few_shots:
        few_shot_block = "## Examples\n" + "\n".join(_format_shot(s) for s in few_shots) + "\n"

    base_prompt = reasoning_nota_prompt["prompt"]
    output_format = reasoning_nota_prompt["output_format"]

    prompt = (
        f"{base_prompt}\n"
        f"{output_format}\n\n"
        f"{few_shot_block}"
        "## Task\n"
        f"Question: {question}\n"
        "Options:\n"
        f"{_format_options_kv(options_for_prompt)}\n"
    )

    info = {
        "test_type": "reasoning_nota",
        "dataset": example.get("dataset", ""),
        "subject_name": example.get("subject_name", ""),
        "split_type": example.get("split_type", ""),
        "correct_index": correct_index,
        "prompt_id": reasoning_nota_prompt.get("id", ""),
    }

    if shuffle_answers:
        info["shuffled"] = True

    return {
        "question": prompt,
        "answer": str(correct_index),  # not used directly; accuracy() uses info["correct_index"]
        "info": info,
    }


def _map_fct_example(
    example: dict[str, Any],
    question: str,
    options_list: list[str],
    correct_index: int | None,
    student_index: int | None,
    num_few_shot: int,
) -> dict[str, Any] | None:
    """
    Map a Med-HALT False Confidence Test (FCT) example.

    Uses the author-provided prompt + output_format + few-shot examples
    (ported into environments/med_halt/prompts.py).

    IMPORTANT:
    - No XML tags.
    - Model should output a single JSON object with keys:
      why_correct, why_others_incorrect, answer, is_answer_correct ("yes"/"no")
    - Few-shot: authors use 2-shot by default; we take the first 2 deterministically.
    """

    # Both indices must exist and be valid
    if correct_index is None or student_index is None:
        return None
    if (
        correct_index < 0
        or student_index < 0
        or correct_index >= len(options_list)
        or student_index >= len(options_list)
    ):
        return None

    proposed_answer = options_list[student_index]
    is_correct = student_index == correct_index

    # Build author-style options dict: numeric keys + "correct answer"
    options_for_prompt: dict[str, Any] = {str(i): opt for i, opt in enumerate(options_list)}
    options_for_prompt["correct answer"] = options_list[correct_index]

    few_shots = reasoning_fct_shots[:num_few_shot]

    def _format_options_kv(opts: dict[str, Any]) -> str:
        # preserve insertion order; do not re-label A/B/C
        return "\n".join(f"{k}: {v}" for k, v in opts.items())

    def _format_shot(shot: dict[str, Any]) -> str:
        inp = shot["input"]
        out = shot["Output"]
        out_json = json.dumps(out, ensure_ascii=False, indent=2)

        return (
            "### Input\n"
            f"Question: {inp['Question']}\n"
            "Options:\n"
            f"{_format_options_kv(inp['Options'])}\n"
            "### Output\n"
            f"{out_json}\n"
        )

    few_shot_block = ""
    if few_shots:
        few_shot_block = "## Examples\n" + "\n".join(_format_shot(s) for s in few_shots) + "\n"

    # ---- assemble final prompt ----
    base_prompt = reasoning_fct_prompt["prompt"]
    output_format = reasoning_fct_prompt["output_format"]

    prompt = (
        f"{base_prompt}\n"
        f"{output_format}\n\n"
        f"{few_shot_block}"
        "## Task\n"
        f"Question: {question}\n"
        "Options:\n"
        f"{_format_options_kv(options_for_prompt)}\n"
        f"The student selected: {proposed_answer}\n"
    )

    return {
        "question": prompt,
        # kept as placeholder; FCT accuracy uses info["is_correct"] + JSON field
        "answer": "1" if is_correct else "0",
        "info": {
            "test_type": "reasoning_FCT",
            "dataset": example.get("dataset", ""),
            "subject_name": example.get("subject_name", ""),
            "split_type": example.get("split_type", ""),
            "proposed_answer": proposed_answer,
            "is_correct": is_correct,
            "correct_index": correct_index,
            "student_index": student_index,
            "prompt_id": reasoning_fct_prompt.get("id", ""),
        },
    }


def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
    """
    Reward function for Med-HALT reasoning.

    Expected model outputs (JSON-only):
      - reasoning_FCT:
          {
            "why_correct": "...",
            "why_others_incorrect": "...",
            "answer": "...",
            "is_answer_correct": "yes" | "no"
          }

      - reasoning_nota:
          {
            "cop": "...",
            "cop_index": <int or numeric string>,
            "why_correct": "...",
            "why_others_incorrect": "..."
          }

    Scoring:
      - FCT: compare is_answer_correct to gold info["is_correct"]
      - NOTA: compare cop_index to gold info["correct_index"]
    """

    if not info:
        return 0.0

    # JSONParser.parse expects a string, but vf-eval gives us Messages (often list[dict])
    if isinstance(completion, str):
        text = completion
    else:
        # JSONParser inherits get_assistant_messages() from Parser
        msgs = parser.get_assistant_messages(completion)
        text = str(msgs[-1]["content"]) if msgs else ""

    data = parser.parse(text, strip=True)

    test_type = info.get("test_type", "")

    if not isinstance(data, dict):
        return 0.0

    # -------------------------
    # FCT: yes/no correctness
    # -------------------------
    if test_type == "reasoning_FCT":
        if "is_correct" not in info:
            return 0.0

        flag = str(data.get("is_answer_correct", "")).strip().lower()
        gold_flag = "yes" if bool(info["is_correct"]) else "no"
        return 1.0 if flag == gold_flag else 0.0

    # -------------------------
    # NOTA: index correctness
    # -------------------------
    if test_type == "reasoning_nota":
        if "correct_index" not in info:
            return 0.0

        cop_index = data.get("cop_index", None)
        try:
            pred_idx = int(cop_index)
        except Exception:
            return 0.0

        try:
            gold_idx = int(info["correct_index"])
        except Exception:
            return 0.0

        return 1.0 if pred_idx == gold_idx else 0.0

    # Unknown test type
    return 0.0


def load_environment(
    use_think: bool = False,
    system_prompt: str | None = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    test_types: list[str] | None = None,
    split_type: str = "val",
    num_few_shot: int = 2,
) -> vf.Environment:
    """
    Load the Med-HALT (Reasoning) environment.

    Args:
        use_think: Enable chain-of-thought reasoning with <think> tags
        system_prompt: Custom system prompt (defaults to standard XML/BOXED prompt)
        shuffle_answers: Randomize the order of answer choices
        shuffle_seed: Random seed for reproducible shuffling
        test_types: List of test types to include (default: ["reasoning_FCT", "reasoning_nota"])
                    Supported: "reasoning_FCT", "reasoning_nota"
        split_type: Logical split to use within HF train split (set to "val" or "train"; defaults to "val" if available)
        num_few_shot: Number of few-shot examples to include (default: 2)

    Returns:
        A SingleTurnEnv configured for Med-HALT reasoning evaluation
    """
    # Default to FCT and nota tests (excluding fake for now as it needs special handling)
    if test_types is None:
        test_types = ["reasoning_FCT", "reasoning_nota"]

    # Validate test types
    invalid_types = [t for t in test_types if t not in REASONING_TEST_TYPES]
    if invalid_types:
        raise ValueError(f"Invalid test types: {invalid_types}. Must be one of: {REASONING_TEST_TYPES}")

    # Load datasets for each test type
    datasets: list[Dataset] = []
    for test_type in test_types:
        # Med-HALT exposes a single HF "train" split and uses a split_type column
        # to mark logical train/val/test. We select the desired logical split here.
        ds_dict = load_dataset("openlifescienceai/Med-HALT", test_type)
        if "train" not in ds_dict:
            # Safety check, but in practice Med-HALT uses train only
            continue

        ds = ds_dict["train"]
        # Filter to the requested logical split (e.g., "val")
        ds = ds.filter(
            lambda ex: ex.get("split_type") == split_type,
        )

        if len(ds) == 0:
            # No examples for this split_type in this test_type
            continue

        # Map the raw examples into the verifiers format
        def _map(ex: dict[str, Any]) -> dict[str, Any] | None:
            return _map_example(
                ex,
                test_type=test_type,
                shuffle_answers=shuffle_answers,
                shuffle_seed=shuffle_seed,
                num_few_shot=num_few_shot,
            )

        # We can safely use HF caching only when the mapping is deterministic
        # w.r.t. shuffle_answers (i.e., when shuffle_answers is False).
        load_from_cache_file = not shuffle_answers
        mapped = ds.map(
            _map,
            remove_columns=ds.column_names,
            load_from_cache_file=load_from_cache_file,
        )

        # Filter out any None / empty questions just in case
        mapped = mapped.filter(
            lambda x: x is not None and x.get("question") is not None,
            load_from_cache_file=load_from_cache_file,
        )

        if len(mapped) > 0:
            datasets.append(mapped)

    if not datasets:
        raise ValueError(f"No valid datasets loaded for test types: {test_types}")

    # Concatenate all test type datasets
    combined_dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    # -------------------------------
    # Parser: raw JSON (FCT + NOTA)
    # -------------------------------
    system_prompt = system_prompt or ""

    parser = JSONParser(
        fields=[
            "is_answer_correct",  # FCT
            "cop_index",  # NOTA
        ],
        extract_fn=lambda x: x,  # (we'll pass it strings)
    )

    # Create rubric with accuracy reward
    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    # Med-HALT only has train split, so we use it as the main dataset
    # No separate eval_dataset
    return vf.SingleTurnEnv(
        dataset=combined_dataset,
        eval_dataset=None,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
