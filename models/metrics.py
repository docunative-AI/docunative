"""
Purpose: 
1. computes token-level F1 score, Recall@3 for retrieval per language accuracy breakdown
2. Tracks NLI label distribution (entailment / contradcition / neutral)
across languages

Author: Docunative team
"""

from pipeline.validate import ParsedOutput


def calculate_f1_score(
    parsed_llm_output: ParsedOutput,
    ground_truth: str    
) -> float:
    """
    Calculating f1 score for token-level inspection.
    Token-level f1 score will be calculated by considering the recall and precision.
    The detailed formula for each of them are shown below:
    1. Recall = total overlapped_tokens / length of ground_truth tokens
    2. Precision = total overlapped_tokens / lenght or llm_answer tokens
    3. F1 score = (2 * precision * recall) / (precision + recall)

    Args:
        - parsed_llm_output: the data parsed with pydantic format from pipeline/validate.py
          parsed_llm_output data structure:
            - answer
            - source_quote
            - parse_success
            - raw_output

    Return:
        token level f1-score
    """
    # Get cleaned answer based on parsed_llm_output data
    llm_answer = parsed_llm_output.answer

    # Split the llm_answer and ground_truth into token-level
    splitted_llm_answer = " ".join(llm_answer.lower().split()).split(" ")
    splitted_ground_truth = " ".join(ground_truth.lower().split()).split(" ")

    # Count overlap tokens between llm_answer and ground_truth
    overlapped_tokens = [token for token in splitted_llm_answer if token in splitted_ground_truth]

    # Calculate recall and precision
    recall = len(overlapped_tokens) / len(splitted_ground_truth)
    precision = len(overlapped_tokens) / len(splitted_llm_answer)

    # Calculate f1 score
    f1_score = (2 * precision * recall) / (precision + recall)

    return round(f1_score, 2)


def calculate_recall_3(
    list_chunks: list,
    ground_truth: str
) -> int:
    """
    Calculating the recall value between llm answer and 3 RAG retrieved chunks.
    Recall value is defined by the existence of ground truth data in list of retrieved chunks.

    Args:
        - list_chunks: list of retrieved chunks which is used by the model for
                       baseline information
        - ground_truth: the actual data which the model should predict

    Return:
        The output must be in the form of int 0 or 1 which means whether the ground truth
        exists onin retrieved chunks or not.

    NOTES:
    Since the ground truth is a part of retrieved chunks, so we can directly check whether
    the ground truth exists on in retrieved chunks by using syntax "in".
    """
    # Check the existence of ground truth in list of chunks
    check_ground_truth = [data for data in list_chunks if data == ground_truth]

    if check_ground_truth:
        # The ground truth data is in list of chunks
        recall_3 = 1
    else:
        # The ground truth data is not in list of chunks
        recall_3 = 0


    return recall_3

    
def nli_label_distribution(
    list_nli_checking_result: list
):
    """
    Track NLI label distribution and calculate the proportion of every NLI classes,
    which are entailment, neutral, and contradiction.

    Args:
        - list_nli_checking_result: list of all answers for each tested language

    Return:
        The percentage (in decimal) of each NLI class for each language
    """
    
    # Collect all of NLI classes within the list result
    list_entailment = [data for data in list_nli_checking_result if data["nli_result"].lower() == "entailment"]
    list_neutral = [data for data in list_nli_checking_result if data["nli_result"].lower() == "neutral"]
    list_contradiction = [data for data in list_nli_checking_result if data["nli_result"].lower() == "contradiction"]

    # Calculate the percentage of each NLI class
    entailment_percentage = round(len(list_entailment) / len(list_nli_checking_result), 2)
    neutral_percentage = round(len(list_neutral) / len(list_nli_checking_result), 2)
    contradiction_percentage = round(len(list_contradiction) / len(list_nli_checking_result), 2)

    return {
        "entailment_percentage": entailment_percentage,
        "neutral_percentage": neutral_percentage,
        "contradiction_percentage": contradiction_percentage
    }


# TESTING
if __name__ == "__main__":
    ## Create dummy data
    print("="*40)
    print("F1 Score calculation testing")
    print("="*40)

    parsed_llm_output = ParsedOutput(
        answer="the passport should have at least one legality checking",
        source_quote="",
        parse_success=True,
        raw_output="answer: the passport should have at least one legality checking"
    )
    ground_truth = "foreigner passport must have at least 1 legality checking"

    f1_score = calculate_f1_score(
        parsed_llm_output,
        ground_truth
    )
    print(f"F1 Score for this llm answer is {f1_score}")
    print("\n\n")

    print("="*40)
    print("Recall@3 calculation testing")
    print("="*40)

    # Define list of chunks
    # Assume every data within this list is a document (simplicity for testing)
    list_chunks = [
        "visitor must provide passport when entering the airport. If they can't provide it, their status will be unknown",
        "passport must contains general information of visitor, such as full name, birthdate, address, etc.",
        "visitor who brings huge amount of money needs to register his/her status in airport legal office."
    ]
    ground_truth = "visitor must provide passport when entering the airport. If they can't provide it, their status will be unknown"
    recall_3 = calculate_recall_3(
        list_chunks,
        ground_truth
    )
    print(f"Recall@3 score is {recall_3}")
    print("\n\n")

    print("="*40)
    print("NLI label distribution checking")
    print("="*40)

    # Define list of NLI result
    list_nli_result = [
        {"llm_answer": "passport should contains a visa for the legality status of visitor",
         "nli_result": "entailment"},
        {"llm_answer": "passport must contains general information of visitor, such as full name, birthdate, address, etc.",
        "nli_result": "neutral"},
        {"llm_answer": "visitor who brings huge amount of money needs to register his/her status in airport legal office.",
         "nli_result": "neutral"},
    ]
    nli_label_distribution_result = nli_label_distribution(
        list_nli_result
    )
    print(f"NLI label distribution result breakdown:\n{nli_label_distribution_result}")




