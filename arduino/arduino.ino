// Pin definitions
#define af7Pin A0
#define tp9Pin A1

void setup()
{
    Serial.begin(9600);
}

void loop()
{
    int af7Value = analogRead(af7Pin);
    int tp9Value = analogRead(tp9Pin);

    // Convert values to range 10 to 100
    float mappedAf7Value = mapValue(af7Value, 0, 1023, 0, 100);
    float mappedTp9Value = mapValue(tp9Value, 0, 1023, 0, 100);

    Serial.print(mappedAf7Value);
    Serial.print("  ,  ");
    Serial.println(mappedTp9Value);

    delay(1); // Wait for 1 second before the next reading
}

// Function to map a value from one range to another
float mapValue(int value, int in_min, int in_max, int out_min, int out_max)
{
    return out_min + (float(value - in_min) * (out_max - out_min) / (in_max - in_min));
}
