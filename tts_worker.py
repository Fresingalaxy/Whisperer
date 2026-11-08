import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("missing text")
    text = sys.argv[1]
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    volume = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    main()
