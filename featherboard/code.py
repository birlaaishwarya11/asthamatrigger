import board
import busio
import adafruit_bme680
import simpleio
import time
import wifi
import socketpool
import ssl
import adafruit_requests
import os


print("Connecting to WiFi...")
wifi.radio.connect(os.getenv("CIRCUITPY_WIFI_SSID"), os.getenv("CIRCUITPY_WIFI_PASSWORD"))
print("Connected! IP:", wifi.radio.ipv4_address)


pool = socketpool.SocketPool(wifi.radio)
requests = adafruit_requests.Session(pool, ssl.create_default_context())


FIREBASE_URL = os.getenv("FIREBASE_URL")


i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c)

while True:
    gas = sensor.gas
    humidity = sensor.humidity
    temp = sensor.temperature
    print("Gas:", gas, "| Humidity:", humidity, "| Temp:", temp)


    poor_gas = gas < 15000
    high_humidity = humidity > 60

    if poor_gas and high_humidity:
        risk = "HIGH"
        simpleio.tone(board.A0, 880, duration=0.2)
        time.sleep(0.1)
        simpleio.tone(board.A0, 880, duration=0.2)
        time.sleep(0.1)
        simpleio.tone(board.A0, 880, duration=0.2)
    elif poor_gas or high_humidity:
        risk = "MODERATE"
        simpleio.tone(board.A0, 660, duration=0.3)
        time.sleep(0.1)
        simpleio.tone(board.A0, 660, duration=0.3)
    else:
        risk = "SAFE"


    data = {
        "gas": gas,
        "humidity": humidity,
        "temperature": temp,
        "risk_level": risk,
        "timestamp_epoch": time.time(),
    }
    try:
        requests.patch(FIREBASE_URL + "/sensor_readings.json", json=data)
        print("Sent to Firebase! Risk:", risk)
    except Exception as e:
        print("Firebase error:", e)

    time.sleep(5)
