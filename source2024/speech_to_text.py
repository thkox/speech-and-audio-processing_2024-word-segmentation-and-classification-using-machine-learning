import io
import os
from tempfile import NamedTemporaryFile
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from source2024 import feature_extraction as fe
from pydub.playback import play

OUTPUT_DIR = 'auxiliary2024/output/results'

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
            except sr.RequestError:
                pass

    return intervals, texts


def calculate_accuracy(ground_truth, predicted):
    """Calculates background, voice, and overall accuracy.

    Handles cases where ground_truth might be empty (all background).

    Args:
        ground_truth: List of tuples representing ground truth intervals.
        predicted: List of tuples representing predicted intervals.

    Returns:
        A tuple containing background, voice, and overall accuracy.
    """

    tp_background = 0
    fp_background = 0
    tp_voice = 0
    fn_voice = 0
    fp_voice = 0
    total_time = 0

    if not ground_truth:  # Handle empty ground truth (all background)
        fp_voice = sum(end - start for start, end in predicted)
        tp_background = max(0, predicted[-1][1] - predicted[0][0] - fp_voice)  # Get total time as background minus predicted voice time
        total_time = tp_background + fp_voice
    else:
        all_intervals = sorted(ground_truth + predicted, key=lambda x: x[0])
        current_state = "background"
        current_start = 0

        for start, end in all_intervals:
            duration = end - start
            total_time += duration
            if (start, end) in ground_truth:
                if current_state == "background":
                    tp_background += duration
                else:
                    tp_voice += duration
                    fn_voice += start - current_start
                current_state = "voice"
                current_start = start
            else:
                if current_state == "background":
                    fp_background += duration
                else:
                    fp_voice += duration
                current_state = "voice"
                current_start = start

        if current_state == "voice" and (current_start, end) in ground_truth:
            tp_voice += end - current_start

    background_accuracy = tp_background / float(total_time) if total_time > 0 else 0.0
    voice_accuracy = tp_voice / float(tp_voice + fn_voice) if tp_voice + fn_voice > 0 else 0.0
    total_correct = tp_background + tp_voice
    total_predicted = total_correct + fp_background + fp_voice
    overall_accuracy = total_correct / float(total_predicted) if total_predicted > 0 else 0.0

    # Convert to percentages with 4 decimal places
    background_accuracy = round(background_accuracy * 100, 4)
    voice_accuracy = round(voice_accuracy * 100, 4)
    overall_accuracy = round(overall_accuracy * 100, 4)

    return background_accuracy, voice_accuracy, overall_accuracy


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

    # Prepare the output file path
    output_file_path = os.path.join(OUTPUT_DIR, f"{title}.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w') as file:
        # compare accuracy with the original intervals
        file.write("Original intervals:\n")
        file.write(f"{intervals_original}\n")
        print("Original intervals:" + str(intervals_original))
        file.write("Predicted intervals:\n")
        file.write(f"{intervals}\n\n")
        print("Predicted intervals:" + str(intervals))

        background_accuracy, voice_accuracy, overall_accuracy = calculate_accuracy(intervals_original, intervals)
        file.write(f"Background accuracy: {background_accuracy}\n")
        print(f"Background accuracy: {background_accuracy}")
        file.write(f"Voice accuracy: {voice_accuracy}\n")
        print(f"Voice accuracy: {voice_accuracy}")
        file.write(f"Overall accuracy: {overall_accuracy}\n\n")
        print(f"Overall accuracy: {overall_accuracy}")

        # Normalize audio to avoid clipping
        audio = audio / np.max(np.abs(audio))

        # Convert to 16-bit PCM (adjust sample_width if needed)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Convert to pydub AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit PCM
            channels=1
        )

        # Play the predicted intervals
        for start, end in intervals:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            segment = audio_segment[start_ms:end_ms]
            file.write(f"Playing interval: {start} to {end} seconds\n")
            print(f"Playing interval: {start} to {end} seconds")
            play(segment)

            # Export the segment to a temporary file
            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                segment.export(temp_file.name, format="wav")
                temp_file_path = temp_file.name

            # Transcribe the temporary file
            segment_intervals, segment_texts = transcribe_audio(temp_file_path)

            # Print the first text segment
            if segment_texts:
                file.write(f"Transcribed Text: {segment_texts[0]}\n\n")
                print(f"Transcribed Text: {segment_texts[0]}\n")
            else:
                file.write("No text transcribed for this segment.\n\n")
                print("No text transcribed for this segment.\n\n")

    print(f"Output saved to {output_file_path}")