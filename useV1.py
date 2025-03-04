# prompt: Code tạo mã sử dụng model đã train xong bên trên và bây giờ tôi muốn sử dụng chúngg

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence, text
import numpy as np

# Load the saved tokenizer
with open('/content/vihsd/tokenizer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = load_model('/content/drive/MyDrive/Colab Notebooks/BTLTUYEN/V3.Text_CNN_model_len150_e60_filter64.h5')

# Define the sequence length used during training
sequence_length = 150

def preprocess_input(text):
    """Preprocesses input text for prediction."""

    # Placeholder for the actual preprocessing steps used during training
    # Replace with your actual preprocess function
    text = text.lower() # Replace with your actual preprocessing
    return text

def predict_text(text):
    """Predicts the label of the given text."""

    # Preprocess the input text
    processed_text = preprocess_input(text)

    # Tokenize and pad the text
    text_sequence = tokenizer.texts_to_sequences([processed_text])  # Tokenize
    text_padded = sequence.pad_sequences(text_sequence, maxlen=sequence_length) # Pad sequence

    # Make the prediction
    prediction = model.predict(text_padded)

    # Get the predicted label
    predicted_label = np.argmax(prediction)

    # Map the predicted label to its meaning
    label_mapping = {0: "clean", 1: "offensive", 2: "hate"} # Update according to your model's output
    predicted_class = label_mapping.get(predicted_label, "unknown")

    return predicted_class

json={
    0: ["Hôm nay trời đẹp quá!",
      "Chúc bạn một ngày mới tốt lành.",
      "Mình rất thích đọc sách.",
      "Bông hoa này thật xinh đẹp.",
      "Cảm ơn bạn đã giúp đỡ mình."
      ],
    1: [
        "DM  mày tao ghét mày rất nhiều thằng ranh  này  ",
        "Mày là thằng náo,  bố xưng cháu"
    ]
    }
# Example usage:
# input_text = json[1][0]
input_text = "Cứ phải chửi chó mới chịu im :)))"
predicted_class = predict_text(input_text)
print(f"Predicted class for '{input_text}': {predicted_class}")
