from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import logging
import speech_recognition as sr
import librosa
import numpy as np
from typing import Dict, List, Any
import time
import soundfile as sf
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhonologicalAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
    def convert_audio_format(self, input_path: str) -> str:
        """Convert audio to WAV format and return path"""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                output_path = temp_wav.name
            
            # Convert using pydub
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            # Fallback: use soundfile
            try:
                audio, sample_rate = librosa.load(input_path, sr=16000, mono=True)
                sf.write(output_path, audio, 16000)
                return output_path
            except Exception as e2:
                logger.error(f"Fallback audio conversion failed: {e2}")
                return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Google Speech Recognition"""
        converted_path = None
        try:
            # Convert to WAV first
            converted_path = self.convert_audio_format(audio_path)
            if not converted_path:
                return ""
            
            with sr.AudioFile(converted_path) as source:
                # Adjust for ambient noise and record
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
                
                # Try Google Speech Recognition
                transcription = self.recognizer.recognize_google(audio)
                return transcription.lower()
                
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.warning(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
        finally:
            # Clean up temporary file
            if converted_path and os.path.exists(converted_path):
                os.unlink(converted_path)
    
    def analyze_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Analyze basic audio features"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            duration = len(y) / sr
            rms_energy = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
            
            # Detect pauses/silence
            frame_length = 1024
            hop_length = 256
            rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            silence_threshold = np.mean(rms_frames) * 0.1
            silent_frames = np.sum(rms_frames < silence_threshold)
            pause_ratio = silent_frames / len(rms_frames)
            
            return {
                'duration': duration,
                'rms_energy': float(rms_energy),
                'spectral_centroid': float(spectral_centroid),
                'zero_crossing_rate': float(zero_crossing_rate),
                'pause_ratio': float(pause_ratio),
                'total_frames': len(rms_frames),
                'silent_frames': int(silent_frames)
            }
        except Exception as e:
            logger.error(f"Audio feature analysis error: {e}")
            return {}
    
    def calculate_accuracy(self, actual: str, expected: str) -> float:
        """Calculate accuracy with fuzzy matching"""
        if not actual:
            return 0.0
        
        actual_clean = actual.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Exact match
        if actual_clean == expected_clean:
            return 1.0
        
        # Partial match based on word overlap
        actual_words = set(actual_clean.split())
        expected_words = set(expected_clean.split())
        
        if not expected_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = actual_words.intersection(expected_words)
        union = actual_words.union(expected_words)
        
        if not union:
            return 0.0
            
        similarity = len(intersection) / len(union)
        
        # Bonus for substring matches
        if expected_clean in actual_clean:
            similarity = max(similarity, 0.7)
        
        return round(similarity, 2)
    
    def detect_phonological_errors(self, actual: str, expected: str, task_type: str) -> List[str]:
        """Detect phonological error patterns"""
        errors = []
        
        if not actual:
            return ['no_speech_detected']
        
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Common error patterns
        if task_type == 'phoneme_deletion':
            if expected_lower not in actual_lower:
                errors.append('incorrect_deletion')
            if len(actual_lower) > len(expected_lower) + 2:
                errors.append('extra_phonemes')
                
        elif task_type == 'phoneme_blending':
            if expected_lower not in actual_lower:
                errors.append('blending_difficulty')
            # Check for segmented production
            if any(sep in actual_lower for sep in ['.', ',', ' and ', ' then ']):
                errors.append('segmented_production')
                
        elif task_type == 'nonword_repetition':
            if actual_lower != expected_lower:
                errors.append('repetition_error')
            # Check for common substitutions
            substitutions = [
                ('blom', 'blob'), ('tekip', 'tepid'), 
                ('strin', 'string'), ('plaff', 'plaft')
            ]
            for wrong, right in substitutions:
                if wrong in expected_lower and right in actual_lower:
                    errors.append('phoneme_substitution')
                    break
        
        # General error patterns
        if len(actual_lower) < len(expected_lower) * 0.7:
            errors.append('truncated_response')
        elif len(actual_lower) > len(expected_lower) * 1.5:
            errors.append('elongated_response')
            
        return errors
    
    def analyze_phonological_task(self, audio_path: str, task_type: str, expected: str, **kwargs) -> Dict[str, Any]:
        """Main analysis function for all phonological tasks"""
        start_time = time.time()
        
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        
        # Analyze audio features
        audio_features = self.analyze_audio_features(audio_path)
        
        # Calculate accuracy
        accuracy_score = self.calculate_accuracy(transcription, expected)
        
        # Detect errors
        error_patterns = self.detect_phonological_errors(transcription, expected, task_type)
        
        # Calculate confidence score
        confidence = accuracy_score
        if audio_features.get('pause_ratio', 0) > 0.3:
            confidence *= 0.8  # Penalize for too many pauses
        if audio_features.get('duration', 0) > 10:
            confidence *= 0.9  # Penalize for very long responses
            
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'task_type': task_type,
            'expected_response': expected,
            'actual_transcription': transcription,
            'accuracy_score': accuracy_score,
            'confidence_score': round(confidence, 2),
            'processing_time_ms': int(processing_time * 1000),
            'error_patterns': error_patterns,
            'audio_features': audio_features,
            'analysis_notes': self.generate_analysis_notes(transcription, expected, error_patterns, accuracy_score)
        }
    
    def generate_analysis_notes(self, transcription: str, expected: str, errors: List[str], accuracy: float) -> str:
        """Generate human-readable analysis notes"""
        if not transcription:
            return "No speech was detected in the recording. Please try again in a quieter environment."
        
        if accuracy >= 0.9:
            return "Excellent performance! The response was clear and accurate."
        elif accuracy >= 0.7:
            return "Good performance with minor variations from the expected response."
        elif accuracy >= 0.5:
            return "Moderate performance. Some difficulty with the phonological task was observed."
        else:
            return "Significant difficulty with the phonological task. Consider additional practice and assessment."
        
        # Add specific notes based on error patterns
        if 'segmented_production' in errors:
            return "The sounds were produced separately rather than blended together smoothly."
        elif 'incorrect_deletion' in errors:
            return "The target phoneme was not correctly deleted from the word."
        
        return "Analysis completed successfully."

# Global analyzer instance
analyzer = PhonologicalAnalyzer()

@app.route('/analyze-phonological', methods=['POST'])
def analyze_phonological():
    """Main endpoint for phonological analysis"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        task_type = request.form.get('task_type')
        expected_response = request.form.get('expected_response')
        
        if not task_type or not expected_response:
            return jsonify({'success': False, 'error': 'Missing task_type or expected_response'}), 400
        
        logger.info(f"Processing {task_type} task, expected: {expected_response}")
        
        # Validate file
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name
        
        try:
            # Perform analysis based on task type
            additional_params = {}
            if task_type == 'phoneme_deletion':
                additional_params['original_word'] = request.form.get('original_word', '')
                additional_params['remove_phoneme'] = request.form.get('remove_phoneme', '')
            elif task_type == 'phoneme_blending':
                phonemes_json = request.form.get('phonemes', '[]')
                additional_params['phonemes'] = json.loads(phonemes_json)
            
            result = analyzer.analyze_phonological_task(
                temp_path, task_type, expected_response, **additional_params
            )
            
            logger.info(f"Analysis completed: {result['accuracy_score']:.2f} accuracy")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Analysis processing error: {e}")
            return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'}), 500
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({'success': False, 'error': 'Server error processing request'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'phonological-analysis',
        'version': '1.0.0',
        'timestamp': time.time()
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Phonological Processing API',
        'endpoints': {
            'analyze': '/analyze-phonological (POST)',
            'health': '/health (GET)'
        },
        'supported_tasks': ['phoneme_deletion', 'phoneme_blending', 'nonword_repetition']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    if debug:
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        from gunicorn.app.base import BaseApplication

        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.application = app
                self.options = options or {}
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        options = {
            'bind': f'0.0.0.0:{port}',
            'workers': 2,
            'worker_class': 'sync',
            'timeout': 120,
            'preload_app': True
        }
        
        StandaloneApplication(app, options).run()
