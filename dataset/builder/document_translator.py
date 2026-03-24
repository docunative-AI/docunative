# -*- coding: utf-8 -*-
"""
dataset/builder/document_translator.py

Optional step: takes documents in one language and translates them into another
Used for obtaining documents in low resource level languages 
Translates documents from data generated in high-mid resource level languages

Output: JSONL files per language in dataset/output/ (no.jsonl, ca.jsonl, gl.jsonl).
"""

import argparse
import logging
from pathlib import Path
import json
from typing import Any

# good for european languages
# from deep_translator import DeeplTranslator
import deepl

logger = logging.getLogger(__name__)

# set up correct authentication key of Deepl account
AUTH_KEY = ""

# Paths
DATASET_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DATA_DIR = DATASET_ROOT / "output"
OUTPUT_DIR = ORIGINAL_DATA_DIR / "translated"

DOCUMENT_FIELD = "document_text"

# limit of chars, currently set to the maximum allowed numer of chars of free account per month
CHAR_NUMBER_LIMIT = 500000
    

def _retrieve_document_data(data)  -> dict[str, Any] | None:
    """Retrieves a document info from the given data record."""
    
    try:
        document_record = json.loads(data)
            
        # verify if the document record has text content
        if DOCUMENT_FIELD not in document_record:
            if "doc_id" in document_record:
                doc_id = document_record["doc_id"]
                logger.exception(f"Error: missing field: {DOCUMENT_FIELD} in the document data entry with id: {doc_id}")
            else:
                logger.exception(f"Error: missing {DOCUMENT_FIELD} field in the document data entry")              
            return None
                
    except json.JSONDecodeError as e:
        logger.exception(f"Error decoding JSON from line: {e}")
        return None
             
    return document_record


def translate_documents(
        original_language: str | None = None,
        target_language: str | None = None
        ) -> None:
    """
    Translates documents from one original language to another language.
    Writes JSONL files to dataset/output/translated (one file per language).
    """

    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    deepl_client = deepl.DeepLClient(AUTH_KEY)
    
    original_data = []
    translated_data = []
    original_text =  []
    number_of_lines_failed = 0
    
    try:
        input_filename = f"{ORIGINAL_DATA_DIR}/{original_language}.jsonl"
        with open(input_filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line:
                    document_data = _retrieve_document_data(line)
                else:
                    number_of_lines_failed += 1
                    continue
                    
                if document_data: 
                    
                    original_data.append(document_data)
                    document_text_content = document_data[DOCUMENT_FIELD]
                    original_text.append(document_text_content)
                else:
                    number_of_lines_failed += 1

        original_documents_text_corpus = "".join(original_text)
        original_documents_text_corpus_char_count = len(original_documents_text_corpus)
        print(f"{original_language}.jsonl document has {original_documents_text_corpus_char_count:,} chars")
        
        # translate only if number of characters is within the limit for open account
        if original_documents_text_corpus_char_count < CHAR_NUMBER_LIMIT:
            for document_data in original_data:
                
                document_text_content = document_data[DOCUMENT_FIELD]
                translated_text = deepl_client.translate_text(
                    document_text_content, 
                    source_lang = original_language, 
                    target_lang = target_language
                    )
                document_data[DOCUMENT_FIELD] = translated_text.text
                
                document_id = document_data["doc_id"]
                document_id_bases = document_id.removeprefix(original_language)
                document_id_new = target_language + document_id_bases
                
                document_data["doc_id"] = document_id_new
                document_data["language"] = target_language
                
                translated_data.append(document_data)
        else:
            logger.warning(f"The number of characters of the text courpus is higher than limit: number of characters = {original_documents_text_corpus_char_count}")
            logger.warning(f"Number of characters = {original_documents_text_corpus_char_count}, limit = {CHAR_NUMBER_THRESHOLD}")
        
        # Save the translated data to the new JSON file
        output_filename = f"{OUTPUT_DIR}/{target_language}.jsonl"
        with open(output_filename, 'w', encoding = 'utf-8') as f:
            for row in translated_data:
                print(row)
                print(type(row))
                f.write(json.dumps(row, ensure_ascii = False) + "\n")
            
        if number_of_lines_failed == 0:
            logger.info(f"Successfully translated '{input_filename}' to '{target_language}' and saved as '{output_filename}'.")
        else: 
            logger.info(f"Translated '{input_filename}' to '{target_language}' and saved as '{output_filename}'.")
            logger.warning(f"Some errors occured: number of lines failed = {number_of_lines_failed}. See previous log for more information")

    except FileNotFoundError:
        logger.exception(f"Error: The file '{input_filename}' was not found.")
    except json.JSONDecodeError:
        logger.exception(f"Error: Could not decode JSON from the file '{input_filename}'.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}, see previous logs")


def main() -> None:
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Translate legal documents from one language to another via DeepL or Gooogle Translate."
    )
    parser.add_argument(
        "--lang_from", 
        metavar = "CODE", 
        type = str, 
        help = "Translate from"
    )
    parser.add_argument(
        "--lang_to", 
        metavar = "CODE", 
        type = str, 
        help = "Translate to"
    )
    args = parser.parse_args()

    translate_documents(
        original_language=args.lang_from,
        target_language=args.lang_to
    )


if __name__ == "__main__":
    main()
