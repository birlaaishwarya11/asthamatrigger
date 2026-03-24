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

# Tones per episode label — distinct from environmental buzzer tones
EPISODE_TONES = {
    "cough":  [(440, 0.15), (440, 0.15), (440, 0.15)],  # 3 low pulses
    "sneeze": [(880, 0.30)],                              # 1 sharp high burst
    "wheeze": [(330, 0.50), (330, 0.50)],                 # 2 slow low pulses
}

def play_episode_alert(label):
    tones = EPISODE_TONES.get(label, [(550, 0.2)])
    for freq, dur in tones:
        simpleio.tone(board.A0, freq, duration=dur)
        time.sleep(0.08)

def check_new_episode(last_epoch):
    """Poll Firebase for the most recently pushed episode. Returns (label, epoch) or None."""
    try:
        resp = requests.get(
            FIREBASE_URL + "/episodes.json?orderBy=%22%24key%22&limitToLast=1"
        )
        data = resp.json()
        if not data:
            return None
        entry = list(data.values())[0]
        ep_epoch = entry.get("episode_start_epoch", 0)
        if ep_epoch > last_epoch:
            return entry.get("label", "cough"), ep_epoch
    except Exception as e:
        print("Episode poll error:", e)
    return None


i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c)

# Ignore any episodes that existed before boot
last_episode_epoch = time.time()

while True:
    gas = sensor.gas
    humidity = sensor.humidity
    temp = sensor.temperature
    print("Gas:", gas, "| Humidity:", humidity, "| Temp:", temp)

    # ── Environmental alert (unchanged) ──────────────────────────────────────
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

    # ── Episode alert ─────────────────────────────────────────────────────────
    result = check_new_episode(last_episode_epoch)
    if result:
        label, ep_epoch = result
        print("Episode detected:", label)
        play_episode_alert(label)
        last_episode_epoch = ep_epoch

    # ── Send sensor reading to Firebase ──────────────────────────────────────
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
