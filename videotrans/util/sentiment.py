import logging
from typing import Optional
from videotrans.configure.config import params, logger

# Emotion mapping for Edge TTS: Chinese and English only
EDGE_EMOTION_MAP = {
    "zh": {"positive": "cheerful", "negative": "sad", "neutral": None},
    "en": {"positive": "cheerful", "negative": "sad", "neutral": None}
}

# Lazy-loaded models
_models = {}

def _load_model(lang: str):
    """Lazy load sentiment model for supported languages only."""
    if lang not in ("zh", "en"):
        raise ValueError(f"Unsupported language for sentiment detection: {lang}")
    if lang in _models:
        return _models[lang]
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        raise ImportError("transformers and torch are required for ML-based sentiment detection. Install with: pip install transformers torch")
    
    if lang == "zh":
        model_name = "uer/chinese-roberta-sentiment"
        device = "cpu"  # Use CPU to avoid GPU dependency
    else:  # en
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        device = "cpu"
    
    logger.debug(f"Loading sentiment model for {lang}: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU
    )
    _models[lang] = classifier
    return classifier

def detect_sentiment(text: str, lang: str) -> Optional[str]:
    """
    Detect sentiment of text and return corresponding Edge TTS emotion style.
    Only supports zh (Chinese) and en (English).
    Returns None for neutral or unsupported languages.
    """
    lang_prefix = lang[:2].lower()
    if lang_prefix not in ("zh", "en"):
        logger.debug(f"Sentiment detection not supported for language: {lang}")
        return None
    
    # Check if emotion is enabled
    if not params.get("edge_tts_emotion_enabled", True):
        return None
    
    try:
        classifier = _load_model(lang_prefix)
    except Exception as e:
        logger.warning(f"Failed to load sentiment model for {lang_prefix}: {e}")
        return None
    
    try:
        # Truncate text to avoid model input limits (512 tokens for most models)
        truncated_text = text[:512] if len(text) > 512 else text
        result = classifier(truncated_text)[0]
        label = result["label"].lower()
        score = result["score"]
        
        # Map model labels to our emotion categories
        if lang_prefix == "zh":
            # uer/chinese-roberta-sentiment labels: positive, negative, neutral
            sentiment = label
        else:  # en
            # distilbert-sst-2 labels: POSITIVE, NEGATIVE
            sentiment = "positive" if "positive" in label else "negative"
        
        # Only apply emotion if confidence is high enough (e.g., >0.7)
        if score < 0.7:
            return None
        
        # Map to Edge TTS emotion
        emotion_map = EDGE_EMOTION_MAP.get(lang_prefix, {})
        edge_emotion = emotion_map.get(sentiment)
        return edge_emotion
    except Exception as e:
        logger.warning(f"Sentiment detection failed for text: {text[:50]}... Error: {e}")
        return None
