# mmai5400A_assignment3
MMAI 5400 Assignment 3 – Aspect-Based Sen- timent Analysis restaurant reviews and determine what the guests thought about the dishes they were served. For this, you’ll need to fine-tune an aspect- based sentiment analysis (ABSA) model to analyze the sentiment expressed about the dishes served


# ABSA_Submission

This project contains the code and fine-tuned model for Aspect-Based Sentiment Analysis (ABSA) on restaurant reviews.

## Directory Structure

- `finetuning.py`: Python script for fine-tuning the SetFit model.
- `absa.py`: Python script containing the `absa` function to perform ABSA on reviews.
- `absa_model/`: Directory containing the fine-tuned model files.
- `reviews.csv`: Dataset used for testing the `absa` function.

## Instructions

1. **Environment Setup:**
   - Create a virtual environment and activate it.
   - Install necessary libraries: `setfit`, `transformers`, `pandas`, `datasets`.

2. **Fine-Tuning the Model:**
   - Run `finetuning.py` to fine-tune the model on the provided dataset.
   - The fine-tuned model will be saved in the `absa_model/` directory.

3. **Performing ABSA:**
   - Use the `absa` function in `absa.py` to analyze reviews.
   - Example usage:
     ```python
     from absa import absa
     reviews = ["the first restaurant review. it can contain multiple sentences.",
                "the second restaurant review",
                ...]
     results = absa(reviews)
     print(results)
     ```

4. **Testing:**
   - Use `reviews.csv` to test the `absa` function and verify the output format.

## Notes

- Ensure all dependencies are installed before running the scripts.
- Follow PEP 8 style guidelines for code formatting.
- Test the `absa` function with sample reviews to verify output format and correctness.
