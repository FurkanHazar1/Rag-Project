import re
from typing import List
import nltk
from nltk.corpus import stopwords
from zemberek import TurkishMorphology, TurkishTokenizer

class QueryCleaner:
    def __init__(self):
        """Creates the necessary objects for Turkish query cleaning."""
        self.morphology = TurkishMorphology.create_with_defaults()
        self.tokenizer = TurkishTokenizer.DEFAULT
        self.stop_words = set(stopwords.words('turkish'))
        
    def clean_query(self, query: str) -> str:
        """
        Cleans a given Turkish query by removing verbs, punctuation, and stop words.
        
        Args:
            query: The Turkish query text to process
            
        Returns:
            Cleaned query text
        """
        # Get tokens and lowercase them
        tokens = [token.content.lower() for token in self.tokenizer.tokenize(query)]
        
        clean_words = []
        
        for word in tokens:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            
            # Skip empty words and stop words
            if not word.strip() or word in self.stop_words:
                continue
            
            # Analyze the word
            if not self.is_word_verb(word):
                clean_words.append(word)
        
        return " ".join(clean_words)
    
    def is_word_verb(self, word: str) -> bool:
        """Checks if a word is a verb."""
        try:
            analysis_results = self.morphology.analyze(word)
            
            # If analysis result is empty, assume it's not a verb
            if not analysis_results:
                return False
                
            for analysis in analysis_results:
                analysis_str = str(analysis)
                if "Verb" in analysis_str or "VERB" in analysis_str:
                    return True
            return False
        except:
            # In case of an analysis error, do not consider it a verb
            return False


