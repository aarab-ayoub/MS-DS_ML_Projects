import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

class SentimentMarkovAnalyzer:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.sentiment_words = {
            'pos': ['excellent', 'incroyable', 'incroyables', 'génial', 'super', 'bon'],
            'neg': ['décevant', 'décevante', 'ennuyeuse', 'mauvais', 'terrible', 'horrible', 'nul'],
            'neu': ['le', 'la', 'les', 'est', 'était', 'sont', 'film', 'histoire', 
                    'acteurs', 'scénario', 'fin', 'musique', 'moyenne', 'correcte', 
                    'mais', 'dans', 'un', 'une']
        }
        self.states = ['pos', 'neu', 'neg']
        self.transition_matrix = None
        
    def get_word_sentiment(self, word: str) -> str:
        """Détermine le sentiment d'un mot."""
        word = word.lower().strip('.,!?;:')
        
        if word in self.sentiment_words['pos']:
            return 'pos'
        elif word in self.sentiment_words['neg']:
            return 'neg'
        else:
            return 'neu'
    
    def train(self, phrases: List[str]):
        """
        Entraîne le modèle en construisant la matrice de transition
        à partir d'un corpus de phrases.
        """
        print("=== Entraînement du modèle ===\n")
        
        # Réinitialiser les transitions
        self.transitions = defaultdict(Counter)
        
        # Analyser chaque phrase
        for phrase in phrases:
            words = phrase.split()
            sentiments = [self.get_word_sentiment(word) for word in words]
            
            print(f"Phrase: {phrase}")
            print(f"Séquence de sentiments: {' → '.join(sentiments)}")
            
            # Compter les transitions
            for i in range(len(sentiments) - 1):
                current = sentiments[i]
                next_state = sentiments[i + 1]
                self.transitions[current][next_state] += 1
            
            print()
        
        # Afficher les transitions
        print("=== Transitions détectées ===")
        all_transitions = []
        for from_state in self.states:
            for to_state in self.states:
                count = self.transitions[from_state][to_state]
                if count > 0:
                    all_transitions.append((from_state, to_state, count))
                    print(f"{from_state.capitalize()} → {to_state.capitalize()}: {count}")
        
        print()
        
        # Construire la matrice de transition
        self._build_transition_matrix()
        
    def _build_transition_matrix(self):
        """Construit la matrice de transition normalisée."""
        matrix = np.zeros((3, 3))
        state_index = {'pos': 0, 'neu': 1, 'neg': 2}
        
        for from_state in self.states:
            total = sum(self.transitions[from_state].values())
            if total > 0:
                for to_state in self.states:
                    count = self.transitions[from_state][to_state]
                    prob = count / total
                    i = state_index[from_state]
                    j = state_index[to_state]
                    matrix[i][j] = prob
        
        self.transition_matrix = matrix
        
    def display_transition_matrix(self):
        """Affiche la matrice de transition de manière formatée."""
        if self.transition_matrix is None:
            print("Le modèle n'a pas encore été entraîné.")
            return
        
        print("=== Matrice de transition ===")
        print("\n         Pos    Neu    Neg")
        print("       " + "-" * 24)
        
        state_labels = ['Pos', 'Neu', 'Neg']
        for i, label in enumerate(state_labels):
            print(f"{label:5} |", end="")
            for j in range(3):
                print(f" {self.transition_matrix[i][j]:5.2f}", end="")
            print()
        print()
    
    def analyze_phrase(self, phrase: str) -> Tuple[List[str], List[Tuple[str, str]], float]:
        """
        Analyse une phrase et calcule la probabilité de la séquence de sentiments.
        
        Returns:
            - Liste des sentiments
            - Liste des transitions
            - Probabilité globale de la séquence
        """
        words = phrase.split()
        sentiments = [self.get_word_sentiment(word) for word in words]
        
        # Calculer les transitions
        transitions = []
        sequence_probability = 1.0
        
        state_index = {'pos': 0, 'neu': 1, 'neg': 2}
        
        for i in range(len(sentiments) - 1):
            current = sentiments[i]
            next_state = sentiments[i + 1]
            transitions.append((current, next_state))
            
            # Probabilité de transition
            if self.transition_matrix is not None:
                i_idx = state_index[current]
                j_idx = state_index[next_state]
                prob = self.transition_matrix[i_idx][j_idx]
                sequence_probability *= prob
        
        return sentiments, transitions, sequence_probability
    
    def sentiment_score(self, sentiments: List[str]) -> float:
        """
        Calcule un score de sentiment global.
        Score entre -1 (très négatif) et +1 (très positif).
        """
        pos_count = sentiments.count('pos')
        neg_count = sentiments.count('neg')
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def predict_next_sentiment(self, current_sentiment: str) -> Dict[str, float]:
        """Prédit la distribution de probabilité du prochain sentiment."""
        if self.transition_matrix is None:
            return {}
        
        state_index = {'pos': 0, 'neu': 1, 'neg': 2}
        idx = state_index[current_sentiment]
        
        predictions = {}
        for i, state in enumerate(self.states):
            predictions[state] = self.transition_matrix[idx][i]
        
        return predictions
    
    def analyze_and_display(self, phrase: str):
        """Analyse complète d'une phrase avec affichage détaillé."""
        print(f"\n{'='*60}")
        print(f"Analyse de la phrase: \"{phrase}\"")
        print('='*60)
        
        words = phrase.split()
        sentiments, transitions, seq_prob = self.analyze_phrase(phrase)
        
        # Décomposition en mots
        print("\n1. Décomposition en mots et sentiments associés:")
        for word, sentiment in zip(words, sentiments):
            word_clean = word.strip('.,!?;:')
            print(f"   \"{word_clean}\" → {sentiment.upper()}")
        
        # Séquence de sentiments
        print(f"\n2. Séquence de sentiments:")
        print(f"   {' → '.join([s.capitalize() for s in sentiments])}")
        
        # Transitions avec probabilités
        print(f"\n3. Transitions et probabilités:")
        state_index = {'pos': 0, 'neu': 1, 'neg': 2}
        for from_s, to_s in transitions:
            if self.transition_matrix is not None:
                i = state_index[from_s]
                j = state_index[to_s]
                prob = self.transition_matrix[i][j]
                print(f"   {from_s.capitalize()} → {to_s.capitalize()}: {prob:.2f}")
        
        # Probabilité globale
        print(f"\n4. Probabilité globale de la séquence:")
        prob_calculation = " × ".join([f"{self.transition_matrix[state_index[t[0]]][state_index[t[1]]]:.2f}" 
                                        for t in transitions])
        print(f"   {prob_calculation} ≈ {seq_prob:.4f}")
        
        # Score de sentiment
        score = self.sentiment_score(sentiments)
        print(f"\n5. Score de sentiment global: {score:.2f}")
        
        # Conclusion
        print(f"\n6. Conclusion:")
        if seq_prob < 0.05:
            print(f"   La faible probabilité ({seq_prob:.4f}) indique un mélange de sentiments.")
        
        pos_words = [w for w, s in zip(words, sentiments) if s == 'pos']
        neg_words = [w for w, s in zip(words, sentiments) if s == 'neg']
        
        if pos_words:
            print(f"   Composante positive: {', '.join([w.strip('.,!?;:') for w in pos_words])}")
        if neg_words:
            print(f"   Composante négative: {', '.join([w.strip('.,!?;:') for w in neg_words])}")
        
        if score > 0.3:
            sentiment_global = "positif"
        elif score < -0.3:
            sentiment_global = "négatif"
        else:
            sentiment_global = "mitigé/neutre"
        
        print(f"   Sentiment global: {sentiment_global.upper()}")
        print()


def main():
    # Créer l'analyseur
    analyzer = SentimentMarkovAnalyzer()
    
    # Corpus d'entraînement (comme dans les slides)
    training_corpus = [
        "Le film est excellent !",
        "L'histoire était moyenne.",
        "Les acteurs sont incroyables.",
        "Le scénario est décevant.",
        "La fin était ennuyeuse.",
        "La musique était correcte."
    ]
    
    # Entraîner le modèle
    analyzer.train(training_corpus)
    
    # Afficher la matrice de transition
    analyzer.display_transition_matrix()
    
    # Analyser une nouvelle phrase (exemple du cours)
    test_phrase = "Le film était incroyable mais la fin décevante."
    analyzer.analyze_and_display(test_phrase)
    
    # Autres exemples d'analyse
    print("\n" + "="*60)
    print("AUTRES EXEMPLES D'ANALYSE")
    print("="*60)
    
    test_phrases = [
        "Le film est super et génial",
        "Cette histoire était terrible et décevante",
        "Le acting est bon mais le scénario est mauvais",
        "Les acteurs sont incroyables et le film excellent"
    ]
    
    for phrase in test_phrases:
        analyzer.analyze_and_display(phrase)
    
    # Prédiction du prochain sentiment
    print("\n" + "="*60)
    print("PRÉDICTION DU PROCHAIN SENTIMENT")
    print("="*60)
    
    for state in ['pos', 'neu', 'neg']:
        predictions = analyzer.predict_next_sentiment(state)
        print(f"\nÉtat actuel: {state.upper()}")
        print("Probabilités du prochain état:")
        for next_state, prob in predictions.items():
            print(f"  → {next_state.upper()}: {prob:.2f}")


if __name__ == "__main__":
    main()
