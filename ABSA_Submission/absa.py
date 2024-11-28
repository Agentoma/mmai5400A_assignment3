import pandas as pd
from setfit import AbsaModel

# Load the fine-tuned model
model = AbsaModel.from_pretrained(
    "absa_model-aspect",
    "absa_model-polarity",
    spacy_model="en_core_web_sm",
)

def absa(reviews):
    """
    Perform Aspect-Based Sentiment Analysis on a list of reviews.

    Parameters:
    reviews (list of str): List of restaurant reviews.

    Returns:
    pandas.DataFrame: DataFrame with columns 'review_id', 'dish', and 'sentiment'.
    """
    preds = model.predict(reviews)
    results = []
    for review_id, review_preds in enumerate(preds):
        for pred in review_preds:
            results.append({
                'review_id': review_id,
                'dish': pred['span'],
                'sentiment': pred['polarity'],
            })
    df = pd.DataFrame(results)
    return df