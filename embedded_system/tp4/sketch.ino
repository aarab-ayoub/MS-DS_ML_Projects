// --- Déclaration des broches (selon le PDF) ---
const int trigPin = 9;      // Trigger du HC-SR04
const int echoPin = 10;     // Echo du HC-SR04
const int ledPin = 7;       // LED
const int buzzerPin = 8;    // Buzzer
const int potPin = A0;      // Potentiomètre

// --- Variables ---
long duration;
int distance;
int seuilAlerte = 0;        // La distance limite réglée par le potentiomètre
int valPot;                 // Valeur du potentiomètre (ajoutée pour plus de clarté)
int toneFreq;               // Fréquence du buzzer (ajoutée pour une meilleure gestion)

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
  // Initialisation du port série pour voir les mesures
  Serial.begin(9600);
}

void loop() {
  // 1. --- Lecture du Potentiomètre (Réglage du seuil) ---
  valPot = analogRead(potPin); // Lecture de la valeur du potentiomètre
  // Conversion de la valeur (0-1023) en une distance seuil (par ex: 10cm à 200cm)
  seuilAlerte = map(valPot, 0, 1023, 10, 200);

  // 2. --- Mesure de la distance avec le HC-SR04 ---
  // Envoi de l'impulsion ultrason
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Lecture de l'écho
  duration = pulseIn(echoPin, HIGH);

  // Calcul de la distance : d = (temps * 0.034) / 2
  distance = duration * 0.034 / 2;

  // 3. --- Affichage sur le moniteur série ---
  Serial.print("Seuil regle: ");
  Serial.print(seuilAlerte);
  Serial.print(" cm | Distance mesuree: ");
  Serial.print(distance);
  Serial.println(" cm");

  // 4. --- Logique d'Alarme (Partie 2 & 3) ---
  if (distance < seuilAlerte) {
    // OBSTACLE DÉTECTÉ !
    
    // Calcul de la fréquence du buzzer en fonction de la proximité de l'objet
    toneFreq = map(distance, 0, seuilAlerte, 1000, 200); 
    toneFreq = constrain(toneFreq, 200, 1000);

    // Activer le buzzer avec la fréquence calculée
    tone(buzzerPin, toneFreq);
    
    // Faire clignoter la LED
    digitalWrite(ledPin, HIGH);
    delay(60); // Délai légèrement plus long pour un flash plus visible
    digitalWrite(ledPin, LOW);
    
    // Le délai restant dépend de la distance
    int delayTime = map(distance, 0, seuilAlerte, 50, 400); // Changement de la plage de délai
    delay(delayTime);
    noTone(buzzerPin); // Couper le son entre les bips
    
  } else {
    // PAS D'OBSTACLE (Zone sûre)
    digitalWrite(ledPin, LOW);
    noTone(buzzerPin);
    delay(150); // Plus de temps de pause entre les mesures pour éviter trop de clignotements
  }
}
