#include <SD.h>
#include <SPI.h>

// --- System parameters ---
const int TARGET_SAMPLING_RATE = 1000;   // keep 1000 Hz sampling (one record every 1ms)
const int SAMPLE_INTERVAL_US = 1000000 / TARGET_SAMPLING_RATE;
const int VIDEO_FPS = 60;                // video frame rate, used to convert to Frame

// --- Hardware pin settings ---
const int NUM_CHANNELS = 3;
const int emgPins[NUM_CHANNELS] = {14, 15, 16};
const int chipSelect = BUILTIN_SDCARD;
const int BTN_PIN = 2;

// --- LED pins ---
const int LED_RED_PIN = 6;
const int LED_EX1_PIN = 8;
const int LED_EX2_PIN = 10;

enum SystemState {
    STATE_IDLE,
    STATE_RECORDING
};
volatile SystemState currentState = STATE_IDLE;

File dataFile;

// --- Double buffering ---
const int BUFFER_SIZE = 500;
struct EmgRecord {
    unsigned long time_ms;     // milliseconds (ms)
    unsigned long frameNumber; // converted 60FPS frame number
    int raw[NUM_CHANNELS];
    int marker;
};

EmgRecord bufferA[BUFFER_SIZE];
EmgRecord bufferB[BUFFER_SIZE];
EmgRecord* volatile writeBuffer = bufferA;
EmgRecord* volatile readBuffer = bufferB;
volatile int writeIndex = 0;
volatile bool bufferReadyForSD = false;

volatile unsigned long sessionStartTime = 0;
volatile bool isFirstSample = false;

// latest values kept for the Serial Plotter
volatile int latestRaw[NUM_CHANNELS] = {0, 0, 0};

IntervalTimer samplingTimer;

bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// --- Serial output control ---
unsigned long lastSerialPrintTime = 0;
const int SERIAL_PRINT_INTERVAL = 10; // output to the Serial Plotter every 10ms (about 100 FPS)

// ==========================================
// Hardware interrupt service routine (ISR) - fires every 1ms
// ==========================================
void samplingISR() {
    if (currentState == STATE_IDLE) return;

    if (currentState == STATE_RECORDING) {
        EmgRecord& rec = writeBuffer[writeIndex];

        // 1. compute elapsed milliseconds
        unsigned long elapsed_ms = millis() - sessionStartTime;
        rec.time_ms = elapsed_ms;

        // 2. convert to a 60 FPS frame number (+1 Frame every 1/60 s), forced to start at 1
        rec.frameNumber = ((elapsed_ms * VIDEO_FPS) / 1000) + 1;

        // read and store the RAW values, also updating the variables shown on Serial
        for(int ch = 0; ch < NUM_CHANNELS; ch++) {
            int val = analogRead(emgPins[ch]);
            rec.raw[ch] = val;
            latestRaw[ch] = val;
        }

        // handle the Marker
        if (isFirstSample) {
            rec.marker = 1;
            isFirstSample = false;
        } else {
            rec.marker = 0;
        }

        writeIndex++;

        // swap when the buffer is full
        if (writeIndex >= BUFFER_SIZE) {
            EmgRecord* temp = writeBuffer;
            writeBuffer = readBuffer;
            readBuffer = temp;
            writeIndex = 0;
            bufferReadyForSD = true;
        }
    }
}

void errorMode() {
    while(1) {
        digitalWrite(LED_RED_PIN, !digitalRead(LED_RED_PIN));
        digitalWrite(LED_EX1_PIN, !digitalRead(LED_EX1_PIN));
        digitalWrite(LED_EX2_PIN, !digitalRead(LED_EX2_PIN));
        delay(100);
    }
}

// ==========================================
// Main program: Setup
// ==========================================
void setup() {
    Serial.begin(115200);

    pinMode(BTN_PIN, INPUT_PULLUP);
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_EX1_PIN, OUTPUT);
    pinMode(LED_EX2_PIN, OUTPUT);

    // LEDs off initially
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_EX1_PIN, LOW);
    digitalWrite(LED_EX2_PIN, LOW);

    analogReadResolution(10);
    analogReadAveraging(4);

    if (!SD.begin(chipSelect)) errorMode();

    // start the timer: fires every 1ms
    samplingTimer.begin(samplingISR, SAMPLE_INTERVAL_US);

    Serial.print("Sampling started at Hz: ");
    Serial.println(TARGET_SAMPLING_RATE);
    Serial.println("Ready. Waiting for button press...");
}

// ==========================================
// Main program: Loop
// ==========================================
void loop() {
    unsigned long currentMillis = millis();

    // ---------------------------------------------------------
    // Serial Plotter display (non-blocking, so the 1ms interrupt is not affected)
    // ---------------------------------------------------------
    if (currentMillis - lastSerialPrintTime >= SERIAL_PRINT_INTERVAL) {
        lastSerialPrintTime = currentMillis;

        if (currentState == STATE_IDLE) {
            // when idle, read the pins directly for easy debugging and electrode placement
            Serial.print(analogRead(emgPins[0])); Serial.print(",");
            Serial.print(analogRead(emgPins[1])); Serial.print(",");
            Serial.println(analogRead(emgPins[2]));
        } else {
            // when recording, use the latest values stored by the ISR to avoid contending for the ADC
            Serial.print(latestRaw[0]); Serial.print(",");
            Serial.print(latestRaw[1]); Serial.print(",");
            Serial.println(latestRaw[2]);
        }
    }

    // ---------------------------------------------------------
    // Button logic
    // ---------------------------------------------------------
    bool reading = digitalRead(BTN_PIN);
    if (reading != lastButtonState) lastDebounceTime = currentMillis;

    if ((currentMillis - lastDebounceTime) > debounceDelay) {
        static bool buttonPressed = false;
        if (reading == LOW && !buttonPressed) {
            buttonPressed = true;

            if (currentState == STATE_IDLE) {
                // --- start recording ---
                digitalWrite(LED_RED_PIN, HIGH);
                digitalWrite(LED_EX1_PIN, HIGH);
                digitalWrite(LED_EX2_PIN, HIGH);

                dataFile = SD.open("SUBJECT_Raw_DATA.csv", FILE_WRITE);
                if (dataFile) {
                    if (dataFile.size() == 0) {
                        // updated header
                        dataFile.println("Time_ms,Frame,Raw0,Raw1,Raw2,Marker");
                    }
                    writeIndex = 0;
                    bufferReadyForSD = false;
                    isFirstSample = true;

                    // --- record the reference milliseconds at the moment the button is pressed ---
                    sessionStartTime = millis();

                    currentState = STATE_RECORDING;
                    Serial.println("Recording Started!"); // status message
                } else {
                    errorMode();
                }
            }
            else if (currentState == STATE_RECORDING) {
                // --- stop recording ---
                currentState = STATE_IDLE;
                digitalWrite(LED_RED_PIN, LOW);
                digitalWrite(LED_EX1_PIN, LOW);
                digitalWrite(LED_EX2_PIN, LOW);

                if (dataFile) {
                    // write the remaining data
                    for (int i = 0; i < writeIndex; i++) {
                        EmgRecord& rec = writeBuffer[i];
                        if (i == writeIndex - 1) rec.marker = 2;

                        dataFile.print(rec.time_ms); dataFile.print(",");
                        dataFile.print(rec.frameNumber); dataFile.print(",");
                        dataFile.print(rec.raw[0]); dataFile.print(",");
                        dataFile.print(rec.raw[1]); dataFile.print(",");
                        dataFile.print(rec.raw[2]); dataFile.print(",");
                        dataFile.println(rec.marker);
                    }
                    dataFile.close();
                    Serial.println("Stopped. Raw data saved.");
                }
            }
        } else if (reading == HIGH) {
            buttonPressed = false;
        }
    }
    lastButtonState = reading;

    // ---------------------------------------------------------
    // SD card writing
    // ---------------------------------------------------------
    if (currentState == STATE_RECORDING && bufferReadyForSD && dataFile) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            EmgRecord& rec = readBuffer[i];

            dataFile.print(rec.time_ms); dataFile.print(",");
            dataFile.print(rec.frameNumber); dataFile.print(",");
            dataFile.print(rec.raw[0]); dataFile.print(",");
            dataFile.print(rec.raw[1]); dataFile.print(",");
            dataFile.print(rec.raw[2]); dataFile.print(",");
            dataFile.println(rec.marker);
        }
        dataFile.flush();
        bufferReadyForSD = false;
    }
}
