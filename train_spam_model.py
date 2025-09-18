import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpamFeatureExtractor:
    def __init__(self):
        pass
        
    def extract_features(self, texts):
        features = []
        for text in texts:
            feature_dict = self._extract_single_text_features(text)
            features.append(feature_dict)
        return pd.DataFrame(features)
    
    def _extract_single_text_features(self, text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        features = {}
        text_lower = text.lower()
        
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['dollar_count'] = text.count('$')
        features['upper_case_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['all_caps_words'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        
        features['has_url'] = 1 if re.search(r'http[s]?://', text) else 0
        features['mention_count'] = len(re.findall(r'@\w+', text))
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        
        features['suspicious_pattern_count'] = len(re.findall(r'\b\w*[@#$%]\w*\b', text))
        features['repeated_char_groups'] = len(re.findall(r'(.)\1{2,}', text))
        
        features['abusive_density'] = features['repeated_char_groups'] / max(features['word_count'], 1)
        
        return features

class SpamDetectorTrainer:
    def __init__(self, model_path='spam_model.pkl'):
        self.model_path = model_path
        self.feature_extractor = SpamFeatureExtractor()
        self.text_pipeline = None
        
    def load_and_validate_data(self, file_path):
        df = pd.read_csv(file_path)
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if not set(df['label'].unique()).issubset({0, 1}):
            raise ValueError("Labels must be 0 (ham) or 1 (spam)")
        logger.info(f"Loaded dataset with {len(df)} samples")
        return df
            
    def preprocess_texts(self, texts):
        def clean_text(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', text)
            text = ' '.join(text.split())
            return text
        return texts.apply(clean_text)
    
    def train_model(self, df):
        processed_texts = self.preprocess_texts(df['text'])
        feature_df = self.feature_extractor.extract_features(df['text'].tolist())
        
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_texts, df['label'], 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
        
        self.text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words=None,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                token_pattern=r'\b\w+\b'
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                class_weight='balanced',
                C=1.0
            ))
        ])
        
        self.text_pipeline.fit(X_train_text, y_train)
        y_pred_text = self.text_pipeline.predict(X_test_text)
        text_report = classification_report(y_test, y_pred_text, output_dict=True)
        
        logger.info(f"Accuracy: {text_report['accuracy']:.4f}")
        logger.info(f"Spam Precision: {text_report['1']['precision']:.4f}")
        logger.info(f"Spam Recall: {text_report['1']['recall']:.4f}")
        
        self.save_model_components()
        
    def save_model_components(self):
        joblib.dump(self.text_pipeline, f'{self.model_path}_text.pkl')
        joblib.dump(self.feature_extractor, f'{self.model_path}_features.pkl')
        logger.info("Model components saved successfully")

def main():
    trainer = SpamDetectorTrainer('spam_model')
    df = trainer.load_and_validate_data('spam_dataset.csv')
    trainer.train_model(df)
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
