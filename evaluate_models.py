#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_task(task_path: Path) -> dict:
    with task_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_prompt(question: str, choices: list[str]) -> str:
    choice_lines = [f"{chr(65 + idx)}) {choice}" for idx, choice in enumerate(choices)]
    return "\n".join([
        question.strip(),
        "Choices:",
        *choice_lines,
        "Answer:",
    ])


def format_choice(choice: str, index: int) -> str:
    return f"{chr(65 + index)}) {choice}"


def score_choice(model, tokenizer, prompt: str, choice: str, device: torch.device) -> float:
    full_text = f"{prompt} {choice}"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits = model(full_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = full_ids[:, 1:]

    prompt_len = prompt_ids.shape[1]
    choice_log_probs = log_probs[:, prompt_len - 1 :, :]
    choice_target_ids = target_ids[:, prompt_len - 1 :]

    token_log_probs = choice_log_probs.gather(2, choice_target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def evaluate_task(model, tokenizer, task: dict, device: torch.device) -> tuple[float, list[dict]]:
    correct = 0
    total = 0
    answers = []
    for qa in task["qas"]:
        prompt = build_prompt(qa["question"], qa["choices"])
        scores = [
            score_choice(model, tokenizer, prompt, choice, device)
            for choice in qa["choices"]
        ]
        predicted = int(max(range(len(scores)), key=lambda idx: scores[idx]))
        actual = qa["answer_index"]
        answers.append({
            "pred": format_choice(qa["choices"][predicted], predicted),
            "actual": format_choice(qa["choices"][actual], actual),
        })
        if predicted == actual:
            correct += 1
        total += 1
    return (correct / total if total else math.nan), answers


def plot_radar(task_names: list[str], model_results: dict[str, list[float]], output_path: Path) -> None:
    angles = [n / float(len(task_names)) * 2 * math.pi for n in range(len(task_names))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_names)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)

    for model_name, scores in model_results.items():
        values = scores + scores[:1]
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Baby Reasoning Bench tasks.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "BabyLM-community/babylm-baseline-100m-gpt2",
            "BabyLM-community/babylm-baseline-10m-gpt2",
        ],
        help="Model names or paths to evaluate.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("tasks"),
        help="Directory containing task JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_performance_radar.png"),
        help="Path to save the radar chart.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_paths = sorted(args.tasks_dir.glob("*.json"))
    if not task_paths:
        raise SystemExit(f"No task files found in {args.tasks_dir}.")

    tasks = [load_task(task_path) for task_path in task_paths]
    task_names = [task["name"] for task in tasks]

    model_results: dict[str, list[float]] = {}
    model_outputs: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for model_name in args.models:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        scores = []
        task_outputs: dict[str, dict[str, list[dict]]] = {}
        for task in tasks:
            accuracy, answers = evaluate_task(model, tokenizer, task, device)
            scores.append(accuracy)
            task_outputs[task["name"]] = {"answers": answers}
            print(f"{model_name} | {task['name']}: {accuracy:.2%}")
        model_results[model_name] = scores
        model_outputs[model_name] = task_outputs
        print(json.dumps(task_outputs, indent=2))

    plot_radar(task_names, model_results, args.output)
    print(f"Radar chart saved to {args.output}")


if __name__ == "__main__":
    main()
