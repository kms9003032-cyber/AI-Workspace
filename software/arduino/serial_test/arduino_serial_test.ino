String command = "";

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    command = Serial.readStringUntil('\n');

    if (command == "MOVE_X") {
      Serial.println("X Motor Move");
    }

    if (command == "GRIP_CLOSE") {
      Serial.println("Grip Close");
    }
  }
}
