import numpy as np
from collections import defaultdict, Counter
import random

class SimpleLLM:
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.vocab = set()
        self.start_tokens = []
    
    def train(self, corpus):
        """Entraîne le modèle sur un corpus de textes"""
        for text in corpus:
            # Nettoyer et tokenizer
            words = text.lower().split()
            if len(words) < self.order + 1:
                continue
                
            # Stocker les débuts de phrases
            self.start_tokens.append(tuple(words[:self.order]))
            
            # Construire le vocabulaire
            self.vocab.update(words)
            
            # Apprendre les transitions
            for i in range(len(words) - self.order):
                context = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                self.transitions[context][next_word] += 1
    
    def _get_next_word(self, context):
        """Choisit le prochain mot basé sur le contexte"""
        if context in self.transitions:
            words, counts = zip(*self.transitions[context].items())
            probabilities = np.array(counts) / sum(counts)
            return np.random.choice(words, p=probabilities)
        else:
            # Backoff: utiliser un contexte plus court
            if len(context) > 1:
                return self._get_next_word(context[1:])
            else:
                return random.choice(list(self.vocab))
    
    def generate_text(self, max_length=20, start_context=None):
        """Génère du texte avec le modèle"""
        if start_context:
            context = tuple(start_context.lower().split())
            if len(context) != self.order:
                raise ValueError(f"Le contexte doit avoir {self.order} mots")
        else:
            context = random.choice(self.start_tokens)
        
        generated = list(context)
        
        for _ in range(max_length - self.order):
            next_word = self._get_next_word(tuple(generated[-self.order:]))
            generated.append(next_word)
            
            # Arrêter si on atteint un point final
            if next_word in ['.', '!', '?']:
                break
        
        return ' '.join(generated)
    
    def probability(self, sequence):
        """Calcule la probabilité d'une séquence"""
        words = sequence.lower().split()
        if len(words) < self.order + 1:
            return 0.0
        
        prob = 1.0
        for i in range(len(words) - self.order):
            context = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            
            if context in self.transitions:
                total = sum(self.transitions[context].values())
                count = self.transitions[context][next_word]
                prob *= (count / total) if total > 0 else 0.0
            else:
                prob *= 0.0  # Transition jamais vue
        
        return prob

# Exemple d'utilisation
if __name__ == "__main__":
   corpus = [
        "je mange une pomme rouge",
        "je bois un jus d orange", 
        "une pomme est un fruit",
        "un jus est une boisson",
        "je regarde un film intéressant",
        "un film est une œuvre artistique",
        "la pomme est délicieuse",
		"le jus est rafraîchissant",
        "regarder un film est amusant",
        "les fruits sont bons pour la santé"
    ]
	# corpus = [
	# 	# ===== Original sentences =====
	# 	"je mange une pomme rouge",
	# 	"je bois un jus d orange",
	# 	"une pomme est un fruit",
	# 	"un jus est une boisson",
	# 	"je regarde un film intéressant",
	# 	"un film est une œuvre artistique",

	# 	# ===== Daily Life =====
	# 	"je marche dans la rue",
	# 	"je parle avec mon ami",
	# 	"je mange du pain frais",
	# 	"je bois de l eau froide",
	# 	"je lis un livre intéressant",
	# 	"je joue avec mon chien",
	# 	"je prépare un bon repas",
	# 	"je fais du sport le matin",
	# 	"je prends le bus pour aller à l école",
	# 	"je me repose dans ma chambre",

	# 	# ===== Food =====
	# 	"une pomme est très sucrée",
	# 	"une orange contient beaucoup de vitamine c",
	# 	"un jus de pomme est délicieux",
	# 	"je coupe une pomme en deux",
	# 	"je cuisine un plat chaud",
	# 	"je mange une salade verte",
	# 	"je bois un café chaud",
	# 	"je prépare une soupe maison",
	# 	"un fruit est bon pour la santé",

	# 	# ===== Movies & Art =====
	# 	"un film raconte une histoire",
	# 	"je regarde un film chaque soir",
	# 	"un acteur joue un personnage célèbre",
	# 	"je préfère les films d action",
	# 	"ce film est très intéressant",
	# 	"un artiste crée une œuvre originale",
	# 	"la musique du film est magnifique",

	# 	# ===== Home & Objects =====
	# 	"ma maison est très calme",
	# 	"je range ma chambre",
	# 	"je ferme la porte doucement",
	# 	"une lampe éclaire la pièce",
	# 	"je mets la table pour le dîner",
	# 	"je nettoie la cuisine",

	# 	# ===== Technology =====
	# 	"j utilise mon ordinateur portable",
	# 	"je charge mon téléphone",
	# 	"je regarde des vidéos sur internet",
	# 	"je joue à un jeu vidéo amusant"
	# ]
    
llm = SimpleLLM(order=2)
   
llm.train(corpus)


print("=== Génération de texte ===")
for _ in range(3):
	sentence = llm.generate_text(max_length=10)
	print("→", sentence)
print("\n=== Quelques transitions apprises ===")
for context, next_words in list(llm.transitions.items())[:5]:
    print(f"{context} → {dict(next_words)}")

print("\n=== Probabilités d'une séquence ===")
for seq in ["je mange une pomme", "un film est intéressant", "je bois un jus"]:
    prob = llm.probability(seq)
    print(f"P('{seq}') = {prob:.6f}")
