from collections import defaultdict, Counter

class SentimentAnalyzer:
    def __init__(self):
        # Mots de sentiment
        self.sentiment_words = {
            'pos': ['excellent', 'incroyable', 'génial', 'super', 'bon'],
            'neg': ['décevant', 'ennuyeuse', 'mauvais', 'terrible', 'nul'],
            'neu': ['le', 'la', 'les', 'est', 'était', 'sont', 'film', 'histoire', 
                    'acteurs', 'scénario', 'fin', 'musique', 'moyenne', 'correcte', 'mais']
        }
        self.transitions = defaultdict(Counter)
    
    def get_sentiment(self, word):
        """Trouve le sentiment d'un mot."""
        word = word.lower().strip('.,!?')
        
        if word in self.sentiment_words['pos']:
            return 'pos'
        elif word in self.sentiment_words['neg']:
            return 'neg'
        else:
            return 'neu'
    
    def train(self, phrases):
        """Entraîne le modèle avec des phrases."""
        print("=== Entraînement ===\n")
        
        for phrase in phrases:
            words = phrase.split()
            sentiments = [self.get_sentiment(w) for w in words]
            
            print(f"Phrase: {phrase}")
            print(f"Sentiments: {' → '.join(sentiments)}")
            
            # Compter les transitions
            for i in range(len(sentiments) - 1):
                self.transitions[sentiments[i]][sentiments[i+1]] += 1
            print()
        
        # Afficher les transitions
        print("=== Transitions ===")
        for from_s in ['neu', 'pos', 'neg']:
            for to_s in ['neu', 'pos', 'neg']:
                count = self.transitions[from_s][to_s]
                if count > 0:
                    print(f"{from_s} → {to_s}: {count}")
        print()
    
    def get_probability(self, from_state, to_state):
        """Calcule P(to_state | from_state)."""
        total = sum(self.transitions[from_state].values())
        if total == 0:
            return 0
        return self.transitions[from_state][to_state] / total
    
    def analyze(self, phrase):
        """Analyse une phrase."""
        print(f"\n=== Analyse: \"{phrase}\" ===\n")
        
        words = phrase.split()
        sentiments = [self.get_sentiment(w) for w in words]
        
        # 1. Mots et sentiments
        print("1. Mots et sentiments:")
        for word, sent in zip(words, sentiments):
            print(f"   {word.strip('.,!?')} → {sent}")
        
        # 2. Séquence
        print(f"\n2. Séquence: {' → '.join(sentiments)}")
        
        # 3. Probabilités des transitions
        print(f"\n3. Probabilités:")
        prob_total = 1.0
        for i in range(len(sentiments) - 1):
            prob = self.get_probability(sentiments[i], sentiments[i+1])
            print(f"   {sentiments[i]} → {sentiments[i+1]}: {prob:.2f}")
            prob_total *= prob
        
        print(f"\n4. Probabilité totale: {prob_total:.4f}")
        
        # 5. Score
        pos_count = sentiments.count('pos')
        neg_count = sentiments.count('neg')
        
        if pos_count + neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            score = 0
        
        print(f"5. Score: {score:.2f}")
        
        if score > 0.3:
            print("6. Conclusion: POSITIF")
        elif score < -0.3:
            print("6. Conclusion: NÉGATIF")
        else:
            print("6. Conclusion: NEUTRE/MITIGÉ")
        print()


# Utilisation
analyzer = SentimentAnalyzer()

# Corpus d'entraînement
corpus = [
    "Le film est excellent !",
    "L'histoire était moyenne.",
    "Les acteurs sont incroyables.",
    "Le scénario est décevant.",
    "La fin était ennuyeuse.",
    "La musique était correcte."
]

# Entraîner
analyzer.train(corpus)

# Analyser des phrases
phrases_test = [
    "Le film était incroyable mais la fin décevante.",
    "Le film est super et génial",
    "Cette histoire était terrible"
]

for phrase in phrases_test:
    analyzer.analyze(phrase)
