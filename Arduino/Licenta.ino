#include <WiFi.h>
#include <Wire.h>
#include <MPU9250_asukiaaa.h>
#include "MAX30105.h"

const char* ssid = "WF-Diana";
const char* password = "banane1133@#";

WiFiServer wifi_server(8081);
WiFiClient client;

MPU9250_asukiaaa sensor_mpu;
MAX30105 particle_sensor;

float ppg_value = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32 starting...");
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  Wire.begin(21, 22);
  sensor_mpu.setWire(&Wire);
  sensor_mpu.beginAccel();
  sensor_mpu.beginGyro();
  sensor_mpu.beginMag();

  if (!particle_sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not detected!");
    while (true);
  } else {
    particle_sensor.setup(5, 4, 2, 200, 118, 2048);
    particle_sensor.setPulseAmplitudeRed(0xF0);
    particle_sensor.setPulseAmplitudeGreen(0);
    Serial.println("MAX30102 initialized.");
  }

  wifi_server.begin();
  Serial.println("Server started on port 8081");
}

void read_ppg_live() {
  particle_sensor.check();
  if (particle_sensor.available()) {
    ppg_value = (float)particle_sensor.getIR();
    particle_sensor.nextSample();
  }
}

void loop() {
  client = wifi_server.available();
  if (client) {
    Serial.println("Client connected");
    String request = client.readStringUntil('\r');
    client.read();

    if (request.indexOf("GET /data") >= 0) {
      sensor_mpu.accelUpdate();
      sensor_mpu.gyroUpdate();

      read_ppg_live();

      String json = "{";
      json += "\"accelX\":" + String(sensor_mpu.accelX()) + ",";
      json += "\"accelY\":" + String(sensor_mpu.accelY()) + ",";
      json += "\"accelZ\":" + String(sensor_mpu.accelZ()) + ",";
      json += "\"gyroX\":" + String(sensor_mpu.gyroX()) + ",";
      json += "\"gyroY\":" + String(sensor_mpu.gyroY()) + ",";
      json += "\"gyroZ\":" + String(sensor_mpu.gyroZ()) + ",";
      json += "\"ppgRaw\":" + String(ppg_value);
      json += "}";

      client.println("HTTP/1.1 200 OK");
      client.println("Content-Type: application/json");
      client.println("Access-Control-Allow-Origin: *");
      client.println("Connection: close");
      client.println();
      client.println(json);

      Serial.println("Data sent: " + json);
    } else {
      client.println("Error");
    }

    delay(1);
    client.stop();
    Serial.println("Client disconnected\n");
    delay(50);
  }
}
