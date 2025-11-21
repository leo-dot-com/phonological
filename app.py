from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import logging
import librosa
import numpy as np
from typing import Dict, List, Any
import time
import soundfile as sf
from pydub import AudioSegment
import requests
import io

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhonologicalAnalyzer:
    def __init__(self):
        # No recognizer initialization needed for Whisper
        pass
        
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
            logger.info(f"Audio converted to WAV: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None
    
    def transcribe_with_whisper(self, audio_path: str) -> str:
        """Transcribe audio using Whisper API"""
        try:
            # Use the same Whisper service as read_aloud component
            whisper_url = "https://readaloud-production.up.railway.app/transcribe"
            
            with open(audio_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                response = requests.post(whisper_url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    transcription = result.get('transcription', '').strip()
                    logger.info(f"Whisper transcription: '{transcription}'")
                    return transcription.lower()
                else:
                    logger.warning(f"Whisper API returned error: {result.get('error')}")
            else:
                logger.error(f"Whisper API HTTP error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
        
        return ""
    
    def transcribe_with_google_fallback(self, audio_path: str) -> str:
        """Fallback transcription using speech_recognition"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            
            converted_path = self.convert_audio_format(audio_path)
            if not converted_path:
                return ""
            
            with sr.AudioFile(converted_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.record(source)
                
                # Try Google Speech Recognition with shorter timeout
                transcription = recognizer.recognize_google(audio, language='en-US', show_all=False)
                logger.info(f"Google transcription: '{transcription}'")
                return transcription.lower()
                
        except Exception as e:
            logger.warning(f"Google fallback failed: {e}")
            return ""
        finally:
            if converted_path and os.path.exists(converted_path):
                os.unlink(converted_path)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Main transcription with multiple fallbacks"""
        logger.info("Starting audio transcription...")
        
        # First try Whisper (more reliable)
        transcription = self.transcribe_with_whisper(audio_path)
        if transcription:
            return transcription
        
        # Fallback to Google
        logger.info("Whisper failed, trying Google fallback...")
        transcription = self.transcribe_with_google_fallback(audio_path)
        if transcription:
            return transcription
        
        logger.warning("All transcription methods failed")
        return ""
    
    def analyze_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Analyze basic audio features"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            # Basic audio analysis
            rms_energy = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Voice activity detection
            frame_length = 1024
            hop_length = 256
            rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            silence_threshold = np.percentile(rms_frames, 20)  # Use 20th percentile as threshold
            voice_frames = np.sum(rms_frames > silence_threshold)
            voice_ratio = voice_frames / len(rms_frames) if len(rms_frames) > 0 else 0
            
            logger.info(f"Audio features - Duration: {duration:.2f}s, Voice ratio: {voice_ratio:.2f}")
            
            return {
                'duration': duration,
                'rms_energy': float(rms_energy),
                'spectral_centroid': float(spectral_centroid),
                'voice_ratio': float(voice_ratio),
                'has_audio': duration > 0.1 and voice_ratio > 0.1
            }
        except Exception as e:
            logger.error(f"Audio feature analysis error: {e}")
            return {'has_audio': False}
    
    def calculate_accuracy(self, actual: str, expected: str) -> float:
        """Calculate accuracy with fuzzy matching"""
        if not actual:
            return 0.0
        
        actual_clean = actual.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Exact match
        if actual_clean == expected_clean:
            return 1.0
        
        # Remove common filler words and noise
        filler_words = ['um', 'uh', 'ah', 'er', 'like', 'so']
        for filler in filler_words:
            actual_clean = actual_clean.replace(filler, '')
        actual_clean = ' '.join(actual_clean.split())  # Normalize spaces
        
        # Check if expected is contained in actual
        if expected_clean in actual_clean:
            return 0.9
        
        # Word-based similarity for multi-word responses
        actual_words = set(actual_clean.split())
        expected_words = set(expected_clean.split())
        
        if expected_words:
            common_words = actual_words.intersection(expected_words)
            similarity = len(common_words) / len(expected_words)
            return round(similarity, 2)
        
        return 0.0
    
    def detect_phonological_errors(self, actual: str, expected: str, task_type: str) -> List[str]:
        """Detect phonological error patterns"""
        errors = []
        
        if not actual:
            return ['no_speech_detected']
        
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Common error patterns by task type
        if task_type == 'phoneme_deletion':
            if expected_lower not in actual_lower:
                errors.append('incorrect_deletion')
            # Check if they said the original word instead
            original_word = request.form.get('original_word', '').lower()
            if original_word and original_word in actual_lower:
                errors.append('said_original_word')
                
        elif task_type == 'phoneme_blending':
            if expected_lower not in actual_lower:
                errors.append('blending_difficulty')
            # Check for saying phonemes separately
            phonemes = request.form.get('phonemes', '[]')
            try:
                phoneme_list = json.loads(phonemes)
                if any(phoneme in actual_lower for phoneme in phoneme_list):
                    errors.append('said_individual_phonemes')
            except:
                pass
                
        elif task_type == 'nonword_repetition':
            if actual_lower != expected_lower:
                errors.append('repetition_error')
            # Check for real word substitutions
            real_word_substitutions = {
                'blom': ['blob', 'bloom', 'blimp'],
                'tekip': ['tepid', 'take it', 'teacup'],
                'strin': ['string', 'strain', 'strand'],
                'plaff': ['plaft', 'plough', 'flaff'],
                'grommet': ['grommet', 'gromit', 'grommet']
            }
            if expected_lower in real_word_substitutions:
                for sub in real_word_substitutions[expected_lower]:
                    if sub in actual_lower:
                        errors.append('real_word_substitution')
                        break
        
        return errors
    
    def analyze_phonological_task(self, audio_path: str, task_type: str, expected: str, **kwargs) -> Dict[str, Any]:
        """Main analysis function for all phonological tasks"""
        start_time = time.time()
        
        # First, analyze audio features to check if we have valid audio
        audio_features = self.analyze_audio_features(audio_path)
        
        if not audio_features.get('has_audio', False):
            return {
                'success': False,
                'error': 'No audible speech detected. Please check your microphone and try again.',
                'audio_features': audio_features
            }
        
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        
        # If no transcription but we have audio, provide a helpful message
        if not transcription:
            return {
                'success': False,
                'error': 'Speech was detected but could not be understood. Please speak more clearly and try again.',
                'audio_features': audio_features,
                'debug_info': {
                    'duration': audio_features.get('duration', 0),
                    'voice_ratio': audio_features.get('voice_ratio', 0)
                }
            }
        
        # Calculate accuracy
        accuracy_score = self.calculate_accuracy(transcription, expected)
        
        # Detect errors
        error_patterns = self.detect_phonological_errors(transcription, expected, task_type)
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'task_type': task_type,
            'expected_response': expected,
            'actual_transcription': transcription,
            'accuracy_score': accuracy_score,
            'confidence_score': round(accuracy_score, 2),
            'processing_time_ms': int(processing_time * 1000),
            'error_patterns': error_patterns,
            'audio_features': audio_features,
            'analysis_notes': self.generate_analysis_notes(transcription, expected, error_patterns, accuracy_score),
            'debug_info': {
                'transcription_method': 'whisper',
                'audio_duration': audio_features.get('duration', 0)
            }
        }
    
    def generate_analysis_notes(self, transcription: str, expected: str, errors: List[str], accuracy: float) -> str:
        """Generate human-readable analysis notes"""
        if accuracy >= 0.9:
            return "Excellent! Clear and accurate response."
        elif accuracy >= 0.7:
            return "Good attempt with minor variations."
        elif accuracy >= 0.5:
            return "Moderate accuracy. Some phonological difficulty detected."
        else:
            return "Significant difficulty with the task. Additional practice recommended."
        
        # Add specific feedback based on errors
        if 'said_original_word' in errors:
            return "You said the original word instead of deleting the target sound."
        elif 'said_individual_phonemes' in errors:
            return "You said the sounds separately instead of blending them together."
        elif 'real_word_substitution' in errors:
            return "You substituted a real word for the nonsense word."
        
        return "Analysis completed."

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
        
        logger.info(f"Processing {task_type} task, expected: '{expected_response}'")
        
        # Validate file
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        # Check file size
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to beginning
        
        if file_size == 0:
            return jsonify({'success': False, 'error': 'Empty audio file'}), 400
        
        logger.info(f"Audio file size: {file_size} bytes")
        
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
                additional_params['phonemes'] = phonemes_json
            
            result = analyzer.analyze_phonological_task(
                temp_path, task_type, expected_response, **additional_params
            )
            
            if result['success']:
                logger.info(f"Analysis successful: '{result['actual_transcription']}' -> {result['accuracy_score']:.2f} accuracy")
            else:
                logger.warning(f"Analysis failed: {result['error']}")
                
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
        'version': '1.1.0',
        'timestamp': time.time(),
        'features': ['whisper_transcription', 'audio_analysis', 'phonological_error_detection']
    })

@app.route('/debug-audio', methods=['POST'])
def debug_audio():
    """Debug endpoint to check audio file"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400
        
        audio_file = request.files['audio']
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name
        
        try:
            # Analyze audio features
            features = analyzer.analyze_audio_features(temp_path)
            
            # Try to get duration using multiple methods
            try:
                audio = AudioSegment.from_file(temp_path)
                pydub_duration = len(audio) / 1000.0
            except Exception as e:
                pydub_duration = 0
                logger.warning(f"PyDub duration failed: {e}")
            
            return jsonify({
                'file_size': os.path.getsize(temp_path),
                'audio_features': features,
                'pydub_duration': pydub_duration,
                'has_audio': features.get('has_audio', False) or pydub_duration > 0.1
            })
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Phonological Processing API',
        'version': '1.1.0',
        'endpoints': {
            'analyze': '/analyze-phonological (POST)',
            'health': '/health (GET)',
            'debug': '/debug-audio (POST)'
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
