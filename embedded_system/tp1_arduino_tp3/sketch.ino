void setup() {
  pinMode(13, OUTPUT);  
  pinMode(10, OUTPUT);  
  pinMode(8, OUTPUT);  
}

void loop() {
  digitalWrite(13, HIGH);
  delay(1000);
  digitalWrite(13, LOW);
  delay(1000);

  digitalWrite(10, HIGH);
  delay(1500);


  for(int i = 0; i < 5; i++)
  {
    digitalWrite(10, LOW);
    delay(500);
    digitalWrite(10, HIGH);
    delay(500);
    
  }
  digitalWrite(10, LOW);
  delay(1500);

  digitalWrite(8, HIGH);
  delay(1500);
  digitalWrite(8, LOW);
  delay(1500);
}
