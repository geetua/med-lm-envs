"""
Minimal integration test for the MedicationQA environment.

Usage:
    python -m environments.medicationqa.test_medicationqa
"""

import asyncio
import json
import os

from environments.medicationqa.medicationqa import load_environment


# ---------------------------------------------------------------------
# Mock OpenAI-style client (mimics AsyncOpenAI)
# ---------------------------------------------------------------------
class MockClient:
    """Minimal mock client that simulates an OpenAI-style chat interface."""

    def __init__(self):
        self.base_url = "mock://"
        self.api_key = "mock"
        self.chat = self.MockChat()

    class MockChat:
        def __init__(self):
            self.completions = self.MockCompletions()

        class MockCompletions:
            async def create(self, *args, **kwargs):
                """Return a fake completion object shaped like OpenAI ChatCompletion."""
                from openai.types.chat import ChatCompletion, ChatCompletionMessage
                from openai.types.chat.chat_completion import Choice

                message = ChatCompletionMessage(role="assistant", content="This is a mock medication answer.")
                choice = Choice(index=0, message=message, finish_reason="stop")

                return ChatCompletion(
                    id="mock-completion",
                    object="chat.completion",
                    created=1234567890,
                    model=kwargs.get("model", "mock-model"),
                    choices=[choice],
                )


# ---------------------------------------------------------------------
# Core test logic
# ---------------------------------------------------------------------
async def run_basic_eval(num_examples: int = 3):
    """Run a short MedicationQA evaluation with a mock model."""
    env = load_environment(judge_model="gpt-4o")  # uses your JUDGE_API_KEY
    dataset = env.eval_dataset
    print(f"✅ Loaded MedicationQA with {len(dataset)} examples")

    # Run a short evaluation with the mock client
    results = await env.evaluate(
        MockClient(),
        model="mock-model",
        num_examples=num_examples,
    )

    # Compute summary stats
    rewards = results.reward
    avg_reward = sum(r for r in rewards if r is not None) / len(rewards)
    print(f"✅ Ran {len(rewards)} examples, avg reward: {avg_reward:.3f}")

    # Save results to file
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/test_medicationqa.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, reward in enumerate(rewards[:num_examples]):
            ex = dataset[i]
            record = {
                "id": ex["id"],
                "question": ex["question"],
                "reference_answer": ex["answer"],
                "reward": reward,
                "model_completion": results.completion[i][0]["content"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"📝 Results saved to: {output_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    asyncio.run(run_basic_eval())


if __name__ == "__main__":
    main()
