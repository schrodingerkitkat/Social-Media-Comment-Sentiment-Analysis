# Sentiment and Emotion Analysis Tool

This Python script leverages  NLP models and libraries to perform a comprehensive analysis of text comments, focusing on sentiment, emotion, and aspect-based analysis, as well as summarization of text data. It is designed to efficiently process large datasets, providing insights into the underlying sentiments and emotional tones of textual comments.

## Features

- **Sentiment Analysis**: Utilizes the `nlptown/bert-base-multilingual-uncased-sentiment` model for sentiment analysis, providing a nuanced understanding of the sentiment expressed in text comments.
- **Emotion Analysis**: Employs a combination of `TextBlob` and the `opinion_lexicon` from NLTK to identify emotional expressions within the text.
- **Aspect-based Sentiment Analysis**: Leverages `TextBlob` to extract nouns (aspects) and their corresponding sentiment polarity, offering a detailed perspective on specific aspects mentioned in the text.
- **Summarization**: Implements the `t5-small` model for summarizing batches of comments, enabling quick comprehension of the main points expressed across multiple comments.
- **Efficient Data Handling**: Utilizes `pandas` for data manipulation and CSV file operations, ensuring high-performance data processing.

## Dependencies

To run this script, you will need the following libraries:

- pandas
- textblob
- transformers
- torch
- nltk

These can be installed via pip using the following command:

```sh
pip install pandas textblob transformers torch nltk
