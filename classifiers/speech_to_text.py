import io
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from classifiers import feature_extraction as fe
import io


def transcribe_audio(file_path):
    """
    Transcribes the speech in an audio file using Google Speech Recognition API.

    Args:
        file_path: The path to the audio file.

    Returns:
        intervals: A list of tuples representing the intervals of the transcribed speech (start, end).
        texts: A list of strings
    """
    # Initialize recognizer class (for recognizing the speech)
    recognizer = sr.Recognizer()

    # Load your audio file
    audio = AudioSegment.from_file(file_path)

    # Split audio where silence is longer than 400ms and get chunks
    chunks = split_on_silence(audio, min_silence_len=400, silence_thresh=-40)

    # Store intervals and their text
    intervals = []
    texts = []

    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Export chunk to a BytesIO object
        chunk_io = io.BytesIO()
        chunk.export(chunk_io, format="wav")
        chunk_io.seek(0)

        with sr.AudioFile(chunk_io) as source:
            audio_listened = recognizer.record(source)

            try:
                # Recognize the chunk
                text = recognizer.recognize_google(audio_listened)
                start_time = sum(len(chunks[j]) for j in range(i)) / 1000.0  # in seconds
                end_time = start_time + len(chunk) / 1000.0  # in seconds
                intervals.append((start_time, end_time))
                texts.append(text)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                pass

    return intervals, texts


def calculate_interval_accuracy(reference_intervals, transcribed_intervals):
    """
    Calculates the accuracy of voice detection based on reference (ground truth) and transcribed intervals.

    Args:
        reference_intervals: List of tuples representing the true voice segments (start_time, end_time).
        transcribed_intervals: List of tuples representing the predicted voice segments (start_time, end_time).

    Returns:
        float: Accuracy of the transcription as a percentage (0-100).
    """
    reference_total_seconds = sum(end - start for start, end in reference_intervals)
    if reference_total_seconds == 0:  # Handle case where there's no voice in the reference
        return 100.0 if not transcribed_intervals else 0.0

    correctly_detected_seconds = 0

    for ref_start, ref_end in reference_intervals:
        for trans_start, trans_end in transcribed_intervals:
            # Calculate overlap between reference and transcribed intervals
            overlap_start = max(ref_start, trans_start)
            overlap_end = min(ref_end, trans_end)
            overlap = max(0, overlap_end - overlap_start)  # Ensure overlap isn't negative
            correctly_detected_seconds += overlap

    accuracy = (correctly_detected_seconds / reference_total_seconds) * 100.0
    return accuracy


def show_predictions(audio, sample_rate, intervals_original, predictions, frame_rate, title):
    """
    Show the predictions of the voice intervals.

    Args:
        audio: The audio signal.
        sample_rate: The sample rate of the audio signal.
        intervals_original: The original voice intervals.
        predictions: The predicted voice intervals.
        frame_rate: The frame rate of the predictions.
        title: The title of the plot.

    """
    intervals = fe.detect_voice_intervals(predictions, frame_rate)
    fe.plot_audio_with_intervals(audio, sample_rate, intervals, title)

    # compare accuracy with the original intervals
    print("Original intervals:", intervals_original)
    print("Predicted intervals:", intervals)
    accuracy = calculate_interval_accuracy(intervals_original, intervals)
    print("Accuracy:", accuracy)

    fe.plot_audio_with_intervals(audio, sample_rate, intervals_original, title)
