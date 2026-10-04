#include <SD.h>
#include <MTP_Teensy.h>

// --- Hardware pin settings (keeps your original LED pins) ---
const int chipSelect = BUILTIN_SDCARD;
const int LED_RED_PIN = 6;
const int LED_EX1_PIN = 8;
const int LED_EX2_PIN = 10;

// ==========================================
// Error blink mode (triggered when the SD card is not inserted properly)
// ==========================================
void errorMode() {
    while(1) {
        digitalWrite(LED_RED_PIN, !digitalRead(LED_RED_PIN));
        delay(100);
    }
}

// ==========================================
// Main program: Setup
// ==========================================
void setup() {
    Serial.begin(115200);

    // initialize the LEDs
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_EX1_PIN, OUTPUT);
    pinMode(LED_EX2_PIN, OUTPUT);
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_EX1_PIN, LOW);
    digitalWrite(LED_EX2_PIN, LOW);

    // give the Serial Monitor some time to connect (wait at most 3 seconds)
    while (!Serial && millis() < 3000);

    Serial.println("====================================");
    Serial.println("  Teensy 4.1 card-reader-only mode starting...  ");
    Serial.println("====================================");

    // 1. initialize the SD card
    if (!SD.begin(chipSelect)) {
        Serial.println("❌ SD card initialization failed! Check that the card is inserted properly.");
        errorMode();
    }
    Serial.println("✅ SD card mounted!");

    // 2. start MTP
    MTP.begin();
    MTP.addFilesystem(SD, "Teensy_EMG_Data"); // this is the drive name you will see in the computer's file explorer

    Serial.println("✅ MTP service started!");
    Serial.println("👉 You can now open the computer's file explorer and access the CSV files directly.");

    // light one LED to show the system is running normally (change as you like)
    digitalWrite(LED_EX1_PIN, HIGH);
}

// ==========================================
// Main program: Loop
// ==========================================
void loop() {
    // devote everything to handling read/copy requests from the computer
    MTP.loop();
}
