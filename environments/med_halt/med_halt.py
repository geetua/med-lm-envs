"""
Med-HALT (Reasoning) Environment

Implements a verifiers environment for the Med-HALT dataset's reasoning hallucination tests.
The dataset includes three test types:
- reasoning_FCT (False Confidence Test): Evaluates if models can assess proposed answers
- reasoning_nota (None of the Above Test): Tests if models can identify when none of the options are correct
- reasoning_fake (Fake Questions Test): Assesses if models can handle nonsensical questions

Paper: https://arxiv.org/abs/2307.15343
Dataset: https://huggingface.co/datasets/openlifescienceai/Med-HALT
"""

from __future__ import annotations

import ast
from typing import Any, Optional

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

# The three reasoning test types in Med-HALT
REASONING_TEST_TYPES = ["reasoning_FCT", "reasoning_nota"]
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H"]


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


def _build_mcq_prompt(question: str, options: dict[str, str]) -> str:
    """
    Build a multiple-choice question prompt with lettered options.

    Args:
        question: The question text
        options: Dictionary mapping indices to option text

    Returns:
        Formatted MCQ prompt string
    """
    # Sort options by their numeric keys
    sorted_items = sorted(options.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    # Filter out 'correct answer' key if present
    option_items = [(k, v) for k, v in sorted_items if k != "correct answer"]

    # Build the prompt with letter labels
    lines = [f"Question: {question}"]
    for idx, (_, option_text) in enumerate(option_items):
        letter = LETTER_INDICES[idx]
        lines.append(f"{letter}. {option_text}")
    lines.append("Answer:")

    return "\n".join(lines)


def _map_example(
    example: dict[str, Any],
    test_type: str,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = None,
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

    # Skipping reasoning_fake for now (no special support yet)
    if test_type == "reasoning_fake":
        return None

    # ---------- FCT PATH ----------
    # FCT examples have a student_index (student's chosen option).
    # We use this as the source of truth instead of relying on the test_type string.
    if student_index is not None:
        return _map_fct_example(
            example=example,
            question=question,
            options_list=options_list,
            correct_index=correct_index,
            student_index=student_index,
            shuffle_answers=shuffle_answers,
            shuffle_seed=shuffle_seed,
        )

    # ---------- NOTA PATH ----------
    # Everything without a student_index is treated as a classic MCQ (reasoning_nota).
    if correct_index is None or correct_index >= len(options_list):
        return None

    correct_answer = example.get("correct_answer", "")
    answer_letter = LETTER_INDICES[correct_index]

    # Handle answer shuffling
    if shuffle_answers:
        shuffled_options, answer_letter, new_index = randomize_multiple_choice(
            options=options_list,
            answer_choice=correct_index,
            labels=LETTER_INDICES[: len(options_list)],
            seed=shuffle_seed,
            row_id=example.get("id", question),
        )
        options_list = shuffled_options
        correct_index = new_index

    # Build the options dict for the prompt
    options_for_prompt = {str(i): opt for i, opt in enumerate(options_list)}

    prompt = _build_mcq_prompt(question, options_for_prompt)

    return {
        "question": prompt,
        "answer": answer_letter,
        "info": {
            "answer_text": correct_answer or options_list[correct_index],
            "test_type": test_type,
            "dataset": example.get("dataset", ""),
            "subject_name": example.get("subject_name", ""),
            "split_type": example.get("split_type", ""),
            **({"options": options_for_prompt} if shuffle_answers else {}),
        },
    }


def _map_fct_example(
    example: dict[str, Any],
    question: str,
    options_list: list[str],
    correct_index: int | None,
    student_index: int | None,
    shuffle_answers: bool,
    shuffle_seed: int | None,
) -> dict[str, Any] | None:
    """
    Map a Med-HALT False Confidence Test (FCT) example.

    FCT asks the model—acting as a medical teacher—to decide whether a student's
    chosen answer to a multiple-choice question is correct or incorrect.

    We implement this as a binary A/B classification:

        A: The student's answer is correct.
        B: The student's answer is incorrect.

    The model is instructed to respond with just 'A' or 'B' inside <answer> tags.
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

    # The student's proposed answer text
    proposed_answer = options_list[student_index]

    # Whether the student's choice matches the true correct index
    is_correct = student_index == correct_index

    # Build options block for context
    option_lines = [f"{LETTER_INDICES[i]}. {opt}" for i, opt in enumerate(options_list)]
    options_block = "\n".join(option_lines)

    # Construct the FCT prompt (no explicit A/B options in the text; just instructions)
    prompt = (
        "You are a medical teacher reviewing a student's answer to a multiple-choice question.\n\n"
        f"{question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"The student selected:\n"
        f"{LETTER_INDICES[student_index]}. {proposed_answer}\n\n"
        "Decide whether the student's answer is correct or incorrect.\n\n"
        "Respond with only one letter inside <answer> tags:\n"
        "- A if the student's answer is correct.\n"
        "- B if the student's answer is incorrect.\n\n"
        "Answer:"
    )

    # Gold label: A if student is actually correct, else B
    answer_letter = "A" if is_correct else "B"

    return {
        "question": prompt,
        "answer": answer_letter,
        "info": {
            "answer_text": "correct" if is_correct else "incorrect",
            "test_type": "reasoning_FCT",
            "dataset": example.get("dataset", ""),
            "subject_name": example.get("subject_name", ""),
            "split_type": example.get("split_type", ""),
            "proposed_answer": proposed_answer,
            "is_correct": is_correct,
            # No need to include MCQ-style options here; the task is binary.
        },
    }


def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
    """Reward based on shared multiple-choice accuracy grading."""
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    test_types: list[str] | None = None,
    split_type: str = "val",
) -> vf.Environment:
    """
    Load the Med-HALT (Reasoning) environment.

    Args:
        use_think: Enable chain-of-thought reasoning with <think> tags
        system_prompt: Custom system prompt (defaults to standard XML/BOXED prompt)
        shuffle_answers: Randomize the order of answer choices
        shuffle_seed: Random seed for reproducible shuffling
        answer_format: Answer format (XML or BOXED)
        test_types: List of test types to include (default: ["reasoning_FCT", "reasoning_nota"])
                   Available: "reasoning_FCT", "reasoning_nota", "reasoning_fake"
        split_type: Logical split to use within HF train split (set to "val" or "train"; defaults to "val" if available)

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

    # Normalize answer format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    # Set up parser and system prompt based on format
    if answer_format == AnswerFormat.XML:
        system_prompt = system_prompt or (THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT)
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format}")

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
