"""
Preprocessing utilities for CLARITY task.

Handles text cleaning, input template formatting, and tokenization preparation.
"""

from typing import Dict, List, Optional
from datasets import Dataset
import re


def create_input_text(
    question: str,
    interview_question: str,
    interview_answer: str,
    include_full_question: bool = True,
    gpt35_summary: str = None,
    gpt35_prediction: str = None
) -> str:
    """
    Create formatted input text for the model.
    
    Format includes both the extracted sub-question and full interview question
    for context, as recommended in the instructions.
    
    Args:
        question: Extracted single sub-question (what we're classifying for).
        interview_question: Full original question (may have multiple sub-questions).
        interview_answer: The answer text.
        include_full_question: Whether to include interview_question context.
        gpt35_summary: Optional GPT-3.5 generated summary to append.
        gpt35_prediction: Optional GPT-3.5 prediction label to append.
    
    Returns:
        Formatted input string.
    """
    if include_full_question and interview_question != question:
        text = (
            f"Interview question (context): {interview_question}\n"
            f"Target sub-question: {question}\n"
            f"Answer: {interview_answer}"
        )
    else:
        text = f"Question: {question}\nAnswer: {interview_answer}"
    
    # Optionally append GPT-3.5 outputs
    if gpt35_summary:
        text += f"\nGPT-3.5 Summary: {gpt35_summary}"
    
    if gpt35_prediction:
        text += f"\nGPT-3.5 Prediction: {gpt35_prediction}"
    
    return text


def clean_text(text: str) -> str:
    """
    Clean text for processing.
    
    Args:
        text: Raw text string.
    
    Returns:
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any null bytes
    text = text.replace('\x00', '')
    
    return text


def create_question_context(
    question: str,
    interview_question: str,
    question_mode: str = "both"
) -> str:
    """
    Create question context string for pair encoding.
    
    Args:
        question: Extracted single sub-question.
        interview_question: Full original question (may have multiple sub-questions).
        question_mode: How to format question context:
            - "question_only": Use only extracted question
            - "interview_only": Use only full interview question
            - "both": Use both (interview + target)
    
    Returns:
        Formatted question context string.
    """
    if question_mode == "question_only":
        return f"Question: {question}"
    elif question_mode == "interview_only":
        return f"Interview question: {interview_question}"
    elif question_mode == "both":
        if interview_question != question:
            return f"Interview question: {interview_question}\nTarget question: {question}"
        else:
            return f"Question: {question}"
    else:
        raise ValueError(f"Unknown question_mode: {question_mode}")


def prepare_model_inputs(
    dataset: Dataset,
    include_full_question: bool = True,
    text_column: str = "model_input",
    use_gpt35_summary: bool = False,
    use_gpt35_prediction: bool = False
) -> Dataset:
    """
    Prepare model input text from dataset.
    
    Args:
        dataset: HF Dataset with question/answer columns.
        include_full_question: Whether to include full interview_question.
        text_column: Name for the new input column.
        use_gpt35_summary: Whether to append GPT-3.5 summary to input.
        use_gpt35_prediction: Whether to append GPT-3.5 prediction to input.
    
    Returns:
        Dataset with added model_input column.
    """
    def _prepare(example):
        # Clean texts
        question = clean_text(example.get("question", ""))
        interview_question = clean_text(example.get("interview_question", ""))
        interview_answer = clean_text(example.get("interview_answer", ""))
        
        # Get GPT-3.5 fields if requested
        gpt35_summary = None
        gpt35_prediction = None
        
        if use_gpt35_summary:
            gpt35_summary = clean_text(example.get("gpt3.5_summary", ""))
            if not gpt35_summary:
                gpt35_summary = None
        
        if use_gpt35_prediction:
            gpt35_prediction = clean_text(example.get("gpt3.5_prediction", ""))
            if not gpt35_prediction:
                gpt35_prediction = None
        
        # Create formatted input
        example[text_column] = create_input_text(
            question=question,
            interview_question=interview_question,
            interview_answer=interview_answer,
            include_full_question=include_full_question,
            gpt35_summary=gpt35_summary,
            gpt35_prediction=gpt35_prediction
        )
        
        return example
    
    return dataset.map(_prepare)


def prepare_model_inputs_pair(
    dataset: Dataset,
    question_mode: str = "both"
) -> Dataset:
    """
    Prepare model inputs for PAIR encoding.
    
    Instead of concatenating everything into one sequence, this creates:
    - text: question context (sequence A)
    - text_pair: answer (sequence B)
    
    The tokenizer will handle these separately with truncation="only_second"
    to ensure the question is never truncated.
    
    Args:
        dataset: HF Dataset with question/answer columns.
        question_mode: How to format question context:
            - "question_only": Just the extracted question
            - "interview_only": Just the full interview question
            - "both": Both interview question and target question
    
    Returns:
        Dataset with 'text' and 'text_pair' columns.
    """
    def _prepare(example):
        # Clean texts
        question = clean_text(example.get("question", ""))
        interview_question = clean_text(example.get("interview_question", ""))
        interview_answer = clean_text(example.get("interview_answer", ""))
        
        # Create question context (sequence A)
        example["text"] = create_question_context(
            question=question,
            interview_question=interview_question,
            question_mode=question_mode
        )
        
        # Answer is sequence B
        example["text_pair"] = interview_answer
        
        return example
    
    return dataset.map(_prepare)


def get_text_lengths(dataset: Dataset, text_column: str = "model_input") -> Dict:
    """
    Compute text length statistics.
    
    Args:
        dataset: Dataset with text column.
        text_column: Name of the text column to analyze.
    
    Returns:
        Dict with length statistics.
    """
    lengths = [len(text.split()) for text in dataset[text_column]]
    
    return {
        "mean": sum(lengths) / len(lengths),
        "median": sorted(lengths)[len(lengths) // 2],
        "min": min(lengths),
        "max": max(lengths),
        "p50": sorted(lengths)[len(lengths) // 2],
        "p75": sorted(lengths)[int(len(lengths) * 0.75)],
        "p90": sorted(lengths)[int(len(lengths) * 0.90)],
        "p95": sorted(lengths)[int(len(lengths) * 0.95)],
        "p99": sorted(lengths)[int(len(lengths) * 0.99)],
    }


def estimate_truncation_rate(
    dataset: Dataset,
    text_column: str = "model_input",
    max_tokens: int = 512,
    words_per_token: float = 0.75
) -> Dict:
    """
    Estimate what % of examples will be truncated at max_tokens.
    
    Uses a rough heuristic: 1 token ≈ 0.75 words.
    
    Args:
        dataset: Dataset with text.
        text_column: Text column name.
        max_tokens: Maximum token length.
        words_per_token: Approximate words per token ratio.
    
    Returns:
        Dict with truncation statistics.
    """
    word_lengths = [len(text.split()) for text in dataset[text_column]]
    estimated_token_lengths = [wl * words_per_token for wl in word_lengths]
    
    truncated = sum(1 for tl in estimated_token_lengths if tl > max_tokens)
    truncation_rate = truncated / len(estimated_token_lengths)
    
    return {
        "total_examples": len(estimated_token_lengths),
        "truncated_examples": truncated,
        "truncation_rate": truncation_rate,
        "max_tokens": max_tokens,
        "avg_estimated_tokens": sum(estimated_token_lengths) / len(estimated_token_lengths)
    }


def main():
    """Example usage."""
    from load_dataset import load_clarity_dataset, normalize_labels, add_label_ids
    
    print("Loading dataset...")
    dataset = load_clarity_dataset()
    
    print("Preparing train inputs...")
    train = normalize_labels(dataset["train"])
    train = add_label_ids(train)
    train = prepare_model_inputs(train, include_full_question=True)
    
    print("\n" + "="*60)
    print("TEXT LENGTH ANALYSIS")
    print("="*60)
    
    # Show example
    print("\nExample input:")
    print("-" * 60)
    print(train[0]["model_input"][:500] + "...")
    print("-" * 60)
    
    # Length stats
    print("\nLength statistics (word count):")
    stats = get_text_lengths(train)
    for key, value in stats.items():
        print(f"  {key:10s}: {value:8.1f}")
    
    # Truncation estimates
    print("\nTruncation analysis:")
    for max_len in [128, 256, 512, 1024]:
        trunc = estimate_truncation_rate(train, max_tokens=max_len)
        print(f"  max_tokens={max_len:4d}: {trunc['truncation_rate']*100:5.1f}% truncated "
              f"({trunc['truncated_examples']}/{trunc['total_examples']})")
    
    # Answer-only lengths (to see the core issue)
    print("\nAnswer-only length stats:")
    answer_lengths = [len(ans.split()) for ans in train["interview_answer"]]
    print(f"  mean: {sum(answer_lengths)/len(answer_lengths):.1f} words")
    print(f"  max:  {max(answer_lengths)} words")
    answer_over_512_tokens = sum(1 for al in answer_lengths if al * 0.75 > 512)
    print(f"  >512 tokens (est): {answer_over_512_tokens} ({100*answer_over_512_tokens/len(answer_lengths):.1f}%)")


if __name__ == "__main__":
    main()


def prepare_model_inputs_with_evidence(
    dataset: Dataset,
    evidence_k: int = 3,
    include_full_question: bool = True,
    text_column: str = "model_input"
) -> Dataset:
    """
    Prepare model input text with evidence selection.
    
    Applies evidence selection to compress answers before creating input.
    
    Args:
        dataset: HF Dataset with question/answer columns.
        evidence_k: Number of middle sentences to keep in evidence selection.
        include_full_question: Whether to include full interview_question.
        text_column: Name for the new input column.
    
    Returns:
        Dataset with added model_input column using evidence-selected answers.
    """
    from data.evidence_selection import select_evidence
    
    def _prepare(example):
        # Clean texts
        question = clean_text(example.get("question", ""))
        interview_question = clean_text(example.get("interview_question", ""))
        interview_answer = clean_text(example.get("interview_answer", ""))
        
        # Apply evidence selection to the answer
        selected_answer = select_evidence(question, interview_answer, k=evidence_k)
        
        # Create formatted input with selected evidence
        if include_full_question and interview_question != question:
            example[text_column] = (
                f"Interview question (context): {interview_question}\n"
                f"Target sub-question: {question}\n"
                f"Answer (selected): {selected_answer}"
            )
        else:
            example[text_column] = f"Question: {question}\nAnswer (selected): {selected_answer}"
        
        return example
    
    return dataset.map(_prepare)


def create_head_tail_input(
    question: str,
    interview_question: str,
    interview_answer: str,
    tokenizer,
    question_budget: int = 160,
    head_budget: int = 176,
    tail_budget: int = 176
) -> str:
    """
    Create input with head-tail packing strategy.
    
    Allocates fixed token budgets:
    - Question: up to question_budget tokens
    - Answer head: head_budget tokens
    - Answer tail: tail_budget tokens
    
    This preserves beginning and end of answers where evasion
    signals often appear, while removing middle filler.
    """
    # Tokenize question context
    q_context = f"Interview question (context): {interview_question}\nTarget sub-question: {question}\n"
    q_tokens = tokenizer(q_context, truncation=True, max_length=question_budget, add_special_tokens=False)
    q_text = tokenizer.decode(q_tokens["input_ids"], skip_special_tokens=True)
    
    # Tokenize answer
    a_tokens = tokenizer(interview_answer, truncation=False, add_special_tokens=False)
    a_token_ids = a_tokens["input_ids"]
    
    total_budget = head_budget + tail_budget
    
    if len(a_token_ids) <= total_budget:
        # Answer fits, use as-is
        a_text = interview_answer
    else:
        # Split into head and tail
        head_ids = a_token_ids[:head_budget]
        tail_ids = a_token_ids[-tail_budget:]
        
        head_text = tokenizer.decode(head_ids, skip_special_tokens=True)
        tail_text = tokenizer.decode(tail_ids, skip_special_tokens=True)
        
        a_text = f"{head_text} [...] {tail_text}"
    
    return f"{q_text}Answer: {a_text}"


def prepare_model_inputs_head_tail(
    dataset,
    tokenizer,
    question_budget: int = 160,
    head_budget: int = 176,
    tail_budget: int = 176,
    text_column: str = "model_input"
):
    """
    Prepare model inputs with head-tail packing.
    """
    def _prepare(example):
        question = clean_text(example.get("question", ""))
        interview_question = clean_text(example.get("interview_question", ""))
        interview_answer = clean_text(example.get("interview_answer", ""))
        
        example[text_column] = create_head_tail_input(
            question=question,
            interview_question=interview_question,
            interview_answer=interview_answer,
            tokenizer=tokenizer,
            question_budget=question_budget,
            head_budget=head_budget,
            tail_budget=tail_budget
        )
        return example
    
    return dataset.map(_prepare)
