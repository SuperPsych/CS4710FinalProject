def va_to_emotion_from_pred(v, a, th=0.0):
    if v >= th and a >= th:
        return "happy"
    if v >= th and a < th:
        return "calm"
    if v < th and a >= th:
        return "angry"
    return "sad"
