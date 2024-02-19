import pandas as pd
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
import torch
import nltk
from nltk.corpus import opinion_lexicon

nltk.download('opinion_lexicon')

# Initialize transformer model for sentiment and summarization
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

def read_csv(file_path):
    return pd.read_csv(file_path)

def comment_summarization(texts):
    t5_input = t5_tokenizer("summarize: " + " ".join(texts), return_tensors="pt")
    summary_ids = t5_model.generate(t5_input['input_ids'])
    return t5_tokenizer.decode(summary_ids[0])

def aspect_sentiment_analysis(text):
    blob = TextBlob(text)
    return {word: TextBlob(word).sentiment.polarity for word, tag in blob.tags if tag == 'NN'}

def emotion_analysis(text):
    emotion_lexicon = set(opinion_lexicon.positive()) | set(opinion_lexicon.negative())
    words = set(text.lower().split())
    return list(words & emotion_lexicon)

def main():
    file_path = "comments.csv"
    df = read_csv(file_path)

    results = []

    for clip_id in df['clip_id'].unique():
        clip_df = df[df['clip_id'] == clip_id]
        comments = clip_df['comment'].dropna().astype(str).tolist()

        if comments:
            summary = comment_summarization(comments)
            
            tokens = tokenizer(comments, padding=True, truncation=True, max_length=512, return_tensors='pt')
            outputs = model(**tokens)
            sentiment = torch.nn.functional.softmax(outputs.logits, dim=1).mean(dim=0).tolist()

            aspects = aspect_sentiment_analysis(' '.join(comments))
            emotions = emotion_analysis(' '.join(comments))

            results.append({
                'clip_id': clip_id,
                'summary': summary,
                'sentiment': sentiment,
                'aspects': aspects,
                'emotions': emotions
            })
            
    result_df = pd.DataFrame(results)
    print(result_df)
    result_df.to_csv('analyzed_comments.csv', index=False)

if __name__ == "__main__":
    main()
