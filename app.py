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

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhonologicalAnalyzer:
    def __init__(self):
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
            # Fallback: use soundfile
            try:
                audio, sample_rate = librosa.load(input_path, sr=16000, mono=True)
                sf.write(output_path, audio, 16000)
                return output_path
            except Exception as e2:
                logger.error(f"Fallback audio conversion failed: {e2}")
                return None
    
    def transcribe_with_whisper_local(self, audio_path: str) -> str:
        """Transcribe audio using local processing (fallback when Whisper API fails)"""
        try:
            # Convert to WAV first
            wav_path = self.convert_audio_format(audio_path)
            if not wav_path:
                return ""
            
            # Use speech_recognition as fallback
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise - longer duration for better calibration
                recognizer.adjust_for_ambient_noise(source, duration=1.0)
                audio = recognizer.record(source)
                
                # Try Google Speech Recognition with specific parameters for short words
                transcription = recognizer.recognize_google(
                    audio, 
                    language='en-US',
                    show_all=False
                )
                logger.info(f"Local transcription: '{transcription}'")
                return transcription.lower()
                
        except sr.UnknownValueError:
            logger.warning("Local speech recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.warning(f"Local speech recognition error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Local transcription error: {e}")
            return ""
        finally:
            # Clean up temporary file
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Main transcription with local processing only"""
        logger.info("Starting local audio transcription...")
        
        # Use local processing only (more reliable for short words)
        transcription = self.transcribe_with_whisper_local(audio_path)
        if transcription:
            return transcription
        
        logger.warning("Local transcription failed")
        return ""
    
    def analyze_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Analyze basic audio features with proper type conversion"""
        try:
            # Load audio with error handling
            try:
                y, sr = librosa.load(audio_path, sr=16000)
            except Exception as e:
                logger.warning(f"Librosa load failed, trying soundfile: {e}")
                y, sr = sf.read(audio_path)
                if sr != 16000:
                    import resampy
                    y = resampy.resample(y, sr, 16000)
                    sr = 16000
            
            duration = len(y) / sr
            
            # Basic audio analysis with error handling
            try:
                rms_energy = float(np.mean(librosa.feature.rms(y=y)))
            except:
                rms_energy = 0.0
                
            try:
                spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            except:
                spectral_centroid = 0.0
            
            # Voice activity detection
            frame_length = 1024
            hop_length = 256
            try:
                rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                if len(rms_frames) > 0:
                    silence_threshold = np.percentile(rms_frames, 20)
                    voice_frames = np.sum(rms_frames > silence_threshold)
                    voice_ratio = float(voice_frames / len(rms_frames))
                else:
                    voice_ratio = 0.0
            except:
                voice_ratio = 0.0
            
            # Convert to Python native types
            has_audio = bool(duration > 0.1 and voice_ratio > 0.1)
            
            logger.info(f"Audio features - Duration: {duration:.2f}s, Voice ratio: {voice_ratio:.2f}")
            
            return {
                'duration': float(duration),
                'rms_energy': rms_energy,
                'spectral_centroid': spectral_centroid,
                'voice_ratio': voice_ratio,
                'has_audio': has_audio
            }
        except Exception as e:
            logger.error(f"Audio feature analysis error: {e}")
            return {
                'duration': 0.0,
                'rms_energy': 0.0,
                'spectral_centroid': 0.0,
                'voice_ratio': 0.0,
                'has_audio': False
            }
    
    def calculate_accuracy(self, actual: str, expected: str) -> float:
        """Calculate accuracy with fuzzy matching for short words"""
        if not actual:
            return 0.0
        
        actual_clean = actual.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Exact match
        if actual_clean == expected_clean:
            return 1.0
        
        # For very short responses (single words), use exact matching
        if len(expected_clean) <= 3:
            if actual_clean == expected_clean:
                return 1.0
            else:
                # Check for common mishearings
                common_mishearings = {
                    'at': ['add', 'ad', 'hat', 'that', 'it'],
                    'top': ['tap', 'tape', 'stop', 'toe'],
                    'red': ['read', 'raid', 'rid', 'road'],
                    'lap': ['lab', 'lamp', 'lip', 'laptop'],
                    'mile': ['mail', 'male', 'smile', 'while']
                }
                if expected_clean in common_mishearings and actual_clean in common_mishearings[expected_clean]:
                    return 0.5
                return 0.0
        
        # Remove common filler words and noise
        filler_words = ['um', 'uh', 'ah', 'er', 'like', 'so', 'the', 'a']
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
    
    def detect_phonological_errors(self, actual: str, expected: str, task_type: str, original_word: str = "") -> List[str]:
        """Detect phonological error patterns"""
        errors = []
        
        if not actual:
            return ['no_speech_detected']
        
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        original_lower = original_word.lower()
        
        # Common error patterns by task type
        if task_type == 'phoneme_deletion':
            if expected_lower not in actual_lower:
                errors.append('incorrect_deletion')
            # Check if they said the original word instead
            if original_lower and original_lower in actual_lower:
                errors.append('said_original_word')
                
        elif task_type == 'phoneme_blending':
            if expected_lower not in actual_lower:
                errors.append('blending_difficulty')
            # Check for saying phonemes separately
            phonemes = request.form.get('phonemes', '[]')
            try:
                phoneme_list = json.loads(phonemes)
                separate_count = sum(1 for phoneme in phoneme_list if phoneme in actual_lower)
                if separate_count >= 2:
                    errors.append('said_individual_phonemes')
            except:
                pass
                
        elif task_type == 'nonword_repetition':
            if actual_lower != expected_lower:
                errors.append('repetition_error')
            # Check for real word substitutions
            real_word_substitutions = {
                'blom': ['blob', 'bloom', 'blimp', 'bomb'],
                'tekip': ['tepid', 'take it', 'teacup', 'take'],
                'strin': ['string', 'strain', 'strand', 'sting'],
                'plaff': ['plaft', 'plough', 'flaff', 'fluff'],
                'grommet': ['gromit', 'grommet', 'comet', 'prompt']
            }
            if expected_lower in real_word_substitutions:
                for sub in real_word_substitutions[expected_lower]:
                    if sub in actual_lower:
                        errors.append('real_word_substitution')
                        break
        
        # General error patterns
        if len(actual_lower) < len(expected_lower) * 0.5:
            errors.append('truncated_response')
        elif len(actual_lower) > len(expected_lower) * 2:
            errors.append('elongated_response')
            
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
        
        # Transcribe audio using local processing only
        transcription = self.transcribe_audio(audio_path)
        
        # If no transcription but we have audio, try to provide basic analysis
        if not transcription:
            # Provide analysis based on audio features only
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'task_type': task_type,
                'expected_response': expected,
                'actual_transcription': 'speech_detected_but_not_understood',
                'accuracy_score': 0.0,
                'confidence_score': 0.0,
                'processing_time_ms': int(processing_time * 1000),
                'error_patterns': ['speech_not_understood'],
                'audio_features': audio_features,
                'analysis_notes': 'Speech was detected but could not be transcribed. This could be due to background noise, unclear pronunciation, or audio quality issues.',
                'debug_info': {
                    'transcription_method': 'local_fallback',
                    'audio_duration': audio_features.get('duration', 0),
                    'voice_ratio': audio_features.get('voice_ratio', 0)
                }
            }
        
        # Calculate accuracy
        accuracy_score = self.calculate_accuracy(transcription, expected)
        
        # Detect errors
        original_word = kwargs.get('original_word', '')
        error_patterns = self.detect_phonological_errors(transcription, expected, task_type, original_word)
        
        processing_time = time.time() - start_time
        
        # Ensure all values are JSON serializable
        result = {
            'success': True,
            'task_type': task_type,
            'expected_response': expected,
            'actual_transcription': transcription,
            'accuracy_score': float(accuracy_score),
            'confidence_score': float(accuracy_score),
            'processing_time_ms': int(processing_time * 1000),
            'error_patterns': error_patterns,
            'audio_features': audio_features,
            'analysis_notes': self.generate_analysis_notes(transcription, expected, error_patterns, accuracy_score),
            'debug_info': {
                'transcription_method': 'local',
                'audio_duration': float(audio_features.get('duration', 0))
            }
        }
        
        return result
    
    def generate_analysis_notes(self, transcription: str, expected: str, errors: List[str], accuracy: float) -> str:
        """Generate human-readable analysis notes"""
        if not transcription or transcription == 'speech_detected_but_not_understood':
            return "Speech was detected but could not be understood. Please try speaking more clearly in a quiet environment."
        
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
            
            # Ensure all values in result are JSON serializable
            def make_serializable(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                else:
                    return obj
            
            result = make_serializable(result)
            
            if result['success']:
                logger.info(f"Analysis successful: '{result['actual_transcription']}' -> {result['accuracy_score']:.2f} accuracy")
            else:
                logger.warning(f"Analysis failed: {result.get('error', 'Unknown error')}")
                
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
        'version': '1.2.0',
        'timestamp': time.time(),
        'features': ['local_transcription', 'audio_analysis', 'phonological_error_detection']
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
            
            # Convert to serializable types
            features = {k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else 
                       int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else
                       bool(v) if isinstance(v, np.bool_) else v 
                       for k, v in features.items()}
            
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
        'version': '1.2.0',
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
