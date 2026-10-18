
import numpy as np


class SignPredictor:
    def __init__(self, model, label_encoder, confidence_threshold=0.70):
        self.model = model
        self.label_encoder = label_encoder
        self.confidence_threshold = confidence_threshold

    def predict(self, sequence):
        """
        Input shape: (30, 63)
        Returns: (predicted_letter, confidence)
        """
        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.shape != (30, 63):
            raise ValueError(
                f"Expected sequence shape (30, 63), got {sequence.shape}"
            )

        probabilities = self.model.predict(
            np.expand_dims(sequence, axis=0),
            verbose=0,
        )[0]

        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

        if confidence < self.confidence_threshold:
            return "?", confidence

        letter = str(
            self.label_encoder.inverse_transform([class_index])[0]
        ).upper()

        return letter, confidence
