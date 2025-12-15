const int buttonPin = 2;

void setup() {
  pinMode(buttonPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  int etat = digitalRead(buttonPin);
  Serial.print("Etat du bouton : ");
  Serial.println(etat);
  
  delay(200);
}
