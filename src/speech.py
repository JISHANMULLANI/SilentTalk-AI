
import pyttsx3


def speak_text(text):
    text = str(text).strip()

    if not text:
        raise ValueError("There is no text to speak.")

    engine = None

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)

        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)

        engine.say(text)
        engine.runAndWait()
        return True

    except Exception as exc:
        raise RuntimeError(f"Text-to-speech failed: {exc}") from exc

    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
