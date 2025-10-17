"""
Sentence tokenization utilities.
Supports both spaCy (when available) and rule-based approach.
"""
import re
import logging

logger = logging.getLogger("spacy_utils")

# Try to import spaCy (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
    _nlp_cache = {}
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Using rule-based sentence segmentation.")


def get_nlp(lang='en'):
    """
    Get spaCy model for language. Falls back to rule-based if not available.
    
    Args:
        lang: 'en' for English (spaCy model available)
        
    Returns:
        spaCy nlp object or None
    """
    if not SPACY_AVAILABLE:
        return None
        
    global _nlp_cache
    if lang not in _nlp_cache:
        try:
            if lang == 'en':
                _nlp_cache[lang] = spacy.load('en_core_web_sm') # type: ignore
            else:
                # Vietnamese model not officially available
                # Fall back to English model for basic sentence segmentation
                logger.warning(f"No spaCy model for '{lang}', using English model")
                _nlp_cache[lang] = spacy.load('en_core_web_sm') # type: ignore
        except OSError:
            logger.warning(f"spaCy model not found for '{lang}'. Install with: python -m spacy download en_core_web_sm")
            return None
    return _nlp_cache[lang]


def rule_based_sent_tokenize(text: str) -> list:
    """
    Rule-based sentence tokenization for Vietnamese and English.
    Works well for formal documents without requiring spaCy models.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Vietnamese and English sentence endings
    # Include ., !, ?, :, ;, and Vietnamese-specific patterns
    
    # Split on sentence boundaries with lookbehind and lookahead
    # Pattern: sentence ending + space + capital letter or number
    sentences = []
    
    # First, split on obvious boundaries
    # Pattern: . ! ? : ; followed by space and uppercase or number
    pattern = r'(?<=[.!?:;])\s+(?=[A-ZĐ0-9"])'
    
    parts = re.split(pattern, text)
    
    # Further process each part
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Handle multiple sentences within a part
        # Split on . ! ? followed by space and capital
        sub_pattern = r'(?<=[.!?])\s+(?=[A-ZĐ])'
        sub_parts = re.split(sub_pattern, part)
        
        for sub_part in sub_parts:
            sub_part = sub_part.strip()
            if len(sub_part) > 5:  # Ignore very short fragments
                sentences.append(sub_part)
    
    # If no sentences found, return original text as single sentence
    if not sentences:
        return [text]
    
    return sentences


def sent_tokenize(text: str, lang: str = 'vi', use_spacy: bool = False) -> list:
    """
    Tokenize text into sentences.
    Uses spaCy if available and requested, otherwise uses rule-based approach.
    
    Args:
        text: Input text
        lang: Language code ('vi' or 'en')
        use_spacy: Force spaCy usage if available (default: False for better compatibility)
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Rule-based is more reliable for Vietnamese documents
    # and doesn't require model downloads
    if not use_spacy or not SPACY_AVAILABLE:
        return rule_based_sent_tokenize(text)
    
    # Try spaCy if explicitly requested
    try:
        nlp = get_nlp(lang if lang == 'en' else 'en')
        if nlp is None:
            return rule_based_sent_tokenize(text)
            
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # If spaCy returns nothing or very few, fall back to rule-based
        if not sentences or len(sentences) < 2:
            return rule_based_sent_tokenize(text)
            
        return sentences
    except Exception as e:
        logger.warning(f"spaCy tokenization failed: {e}. Using rule-based.")
        return rule_based_sent_tokenize(text)
