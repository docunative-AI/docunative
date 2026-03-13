import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def nli_validation(list_premises: list,
                   llm_answer: str):
    """
    This function is used to evalute the logical relationship between
    RAG chunks and llm answer.

    The output will be:
    1. Contradiction
    2. Neutral
    3. Entailment
    """
    # Define model's name
    model_name = "microsoft/deberta-base-mnli"

    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Create output mapping
    output_mapping = {
        0: "contradiction",
        1: "neutral",
        2: "entailment"
    }

    # Tokenization
    result = []
    for premise in list_premises:
        tokenized_data = tokenizer(
            str(premise),
            str(llm_answer),
            return_tensors="pt"
        ).to(model.device)

        # Model prediction
        with torch.no_grad():
            logits = model(**tokenized_data).logits
            prediction = logits.argmax().item()
            prediction_str = output_mapping[prediction]
        
        # Append results
        result.append(
            {
                "llm_answer": llm_answer,
                "premise": premise,
                "nli_checking_result": prediction_str
            }
        )

    
    return result