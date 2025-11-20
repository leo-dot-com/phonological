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
import subprocess
import soundfile as sf

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
        
    def convert_audio_format(self, input_path: str, output_path: str):
        """Convert audio to WAV format for better compatibility"""
        try:
            # Use pydub for audio conversion
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_path, format="wav")
            return True
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            # Fallback: use soundfile
            try:
                audio, sr = librosa.load(input_path, sr=16000, mono=True)
                sf.write(output_path, audio, 16000)
                return True
            except Exception as e2:
                logger.error(f"Fallback audio conversion also failed: {e2}")
                return False
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using multiple fallback methods"""
        try:
            # Convert to WAV first for better compatibility
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            if not self.convert_audio_format(audio_path, temp_wav_path):
                return ""
            
            # Try Google Speech Recognition first
            try:
                with sr.AudioFile(temp_wav_path) as source:
                    audio = self.recognizer.record(source)
                    transcription = self.recognizer.recognize_google(audio)
                    return transcription.lower()
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"Google Speech Recognition error: {e}")
            
            # Fallback: Use vosk or other offline method
            # For now, return empty string if Google fails
            return ""
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
        finally:
            # Clean up temporary file
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
    
    def simple_phoneme_analysis(self, audio_path: str, expected: str) -> List[Dict]:
        """Simple phoneme analysis without MFA"""
        try:
            # Load audio for basic analysis
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract basic audio features
            duration = len(y) / sr
            rms_energy = np.mean(librosa.feature.rms(y=y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Simple phoneme-like segmentation based on energy
            frame_length = 1024
            hop_length = 256
            
            # Create pseudo-phoneme segments
            segments = []
            n_frames = len(y) // hop_length
            
            for i in range(min(10, n_frames)):  # Limit to 10 segments for demo
                start_time = i * hop_length / sr
                end_time = min((i + 1) * hop_length / sr, duration)
                
                if i < len(expected):
                    phoneme = expected[i] if i < len(expected) else "?"
                else:
                    phoneme = "?"
                
                segments.append({
                    'phoneme': phoneme,
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'confidence': 0.8  # Placeholder
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Simple phoneme analysis error: {e}")
            return []
    
    def analyze_phoneme_deletion(self, audio_path: str, expected: str, original_word: str, remove_phoneme: str) -> Dict[str, Any]:
        """Analyze phoneme deletion task"""
        start_time = time.time()
        
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        
        # Simple phoneme analysis
        phoneme_alignment = self.simple_phoneme_analysis(audio_path, expected)
        
        # Calculate accuracy metrics
        accuracy_score = self.calculate_accuracy(transcription, expected)
        phoneme_accuracy = self.calculate_phoneme_accuracy(phoneme_alignment, expected)
        
        # Detect error patterns
        error_patterns = self.detect_error_patterns(transcription, expected, 'deletion')
        
        processing_time = time.time() - start_time
        
        return {
            'actual_transcription': transcription,
            'accuracy_score': accuracy_score,
            'phoneme_accuracy': phoneme_accuracy,
            'phoneme_alignment': phoneme_alignment,
            'error_patterns': error_patterns,
            'processing_time_ms': int(processing_time * 1000),
            'task_type': 'phoneme_deletion',
            'analysis_method': 'basic_audio_analysis'
        }
    
    def analyze_phoneme_blending(self, audio_path: str, expected: str, phonemes: List[str]) -> Dict[str, Any]:
        """Analyze phoneme blending task"""
        start_time = time.time()
        
        transcription = self.transcribe_audio(audio_path)
        phoneme_alignment = self.simple_phoneme_analysis(audio_path, expected)
        
        accuracy_score = self.calculate_accuracy(transcription, expected)
        phoneme_accuracy = self.calculate_phoneme_accuracy(phoneme_alignment, expected)
        error_patterns = self.detect_error_patterns(transcription, expected, 'blending')
        
        processing_time = time.time() - start_time
        
        return {
            'actual_transcription': transcription,
            'accuracy_score': accuracy_score,
            'phoneme_accuracy': phoneme_accuracy,
            'phoneme_alignment': phoneme_alignment,
            'error_patterns': error_patterns,
            'processing_time_ms': int(processing_time * 1000),
            'task_type': 'phoneme_blending',
            'analysis_method': 'basic_audio_analysis'
        }
    
    def analyze_nonword_repetition(self, audio_path: str, expected: str) -> Dict[str, Any]:
        """Analyze nonword repetition task"""
        start_time = time.time()
        
        transcription = self.transcribe_audio(audio_path)
        phoneme_alignment = self.simple_phoneme_analysis(audio_path, expected)
        
        accuracy_score = self.calculate_accuracy(transcription, expected)
        phoneme_accuracy = self.calculate_phoneme_accuracy(phoneme_alignment, expected)
        error_patterns = self.detect_error_patterns(transcription, expected, 'repetition')
        
        processing_time = time.time() - start_time
        
        return {
            'actual_transcription': transcription,
            'accuracy_score': accuracy_score,
            'phoneme_accuracy': phoneme_accuracy,
            'phoneme_alignment': phoneme_alignment,
            'error_patterns': error_patterns,
            'processing_time_ms': int(processing_time * 1000),
            'task_type': 'nonword_repetition',
            'analysis_method': 'basic_audio_analysis'
        }
    
    def calculate_accuracy(self, actual: str, expected: str) -> float:
        """Calculate word-level accuracy with fuzzy matching"""
        if not actual:
            return 0.0
        
        # Simple exact match
        if actual.strip().lower() == expected.strip().lower():
            return 1.0
        
        # Fuzzy matching for partial credit
        actual_words = actual.lower().split()
        expected_words = expected.lower().split()
        
        if len(expected_words) == 0:
            return 0.0
        
        # Calculate similarity based on common words
        common_words = set(actual_words) & set(expected_words)
        similarity = len(common_words) / len(expected_words)
        
        return max(0.0, min(1.0, similarity))
    
    def calculate_phoneme_accuracy(self, alignment: List[Dict], expected: str) -> float:
        """Calculate phoneme-level accuracy"""
        if not alignment:
            return 0.0
        
        # Simple accuracy based on segment count vs expected length
        expected_phonemes = len(expected.replace(' ', ''))
        aligned_phonemes = len(alignment)
        
        if expected_phonemes == 0:
            return 0.0
        
        # Basic ratio - in production, this would use actual phoneme matching
        ratio = min(1.0, aligned_phonemes / expected_phonemes)
        return ratio * 0.8  # Scale down for conservative estimate
    
    def detect_error_patterns(self, actual: str, expected: str, task_type: str) -> List[str]:
        """Detect common phonological error patterns"""
        error_patterns = []
        
        if not actual:
            return ['no_speech_detected']
        
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # Common phonological error patterns
        if task_type == 'deletion':
            if expected_lower not in actual_lower:
                error_patterns.append('incorrect_deletion')
            if len(actual_lower) > len(expected_lower):
                error_patterns.append('extra_sounds')
        
        elif task_type == 'blending':
            if expected_lower not in actual_lower:
                error_patterns.append('blending_difficulty')
            # Check for segmentation (saying sounds separately)
            if any(phoneme in actual_lower for phoneme in [' ', '.', ',']):
                error_patterns.append('segmented_production')
        
        elif task_type == 'repetition':
            if expected_lower not in actual_lower:
                error_patterns.append('repetition_error')
            # Check for phoneme substitutions
            if len(actual_lower) == len(expected_lower) and actual_lower != expected_lower:
                error_patterns.append('phoneme_substitution')
        
        # General error patterns
        if len(actual_lower) < len(expected_lower) * 0.5:
            error_patterns.append('truncated_response')
        
        if len(actual_lower) > len(expected_lower) * 1.5:
            error_patterns.append('elongated_response')
        
        return error_patterns

# Global analyzer instance
analyzer = PhonologicalAnalyzer()

@app.route('/analyze-phonological', methods=['POST'])
def analyze_phonological():
    """Main endpoint for phonological analysis"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        task_type = request.form.get('task_type')
        expected_response = request.form.get('expected_response')
        
        if not task_type or not expected_response:
            return jsonify({'error': 'Missing task_type or expected_response'}), 400
        
        logger.info(f"Processing {task_type} task, expected: {expected_response}")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name
        
        try:
            # Route to appropriate analysis function
            if task_type == 'phoneme_deletion':
                original_word = request.form.get('original_word')
                remove_phoneme = request.form.get('remove_phoneme')
                result = analyzer.analyze_phoneme_deletion(
                    temp_path, expected_response, original_word, remove_phoneme
                )
            
            elif task_type == 'phoneme_blending':
                phonemes_json = request.form.get('phonemes')
                phonemes = json.loads(phonemes_json) if phonemes_json else []
                result = analyzer.analyze_phoneme_blending(
                    temp_path, expected_response, phonemes
                )
            
            elif task_type == 'nonword_repetition':
                result = analyzer.analyze_nonword_repetition(
                    temp_path, expected_response
                )
            
            else:
                return jsonify({'error': 'Invalid task type'}), 400
            
            logger.info(f"Analysis completed: {result['accuracy_score']:.2f} accuracy")
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'phonological-analysis',
        'version': '1.0.0',
        'features': ['phoneme_deletion', 'phoneme_blending', 'nonword_repetition']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    if debug:
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # Use gunicorn in production
        import gunicorn.app.base

        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.application = app
                self.options = options or {}
                super().__init__()

            def load_config(self):
                for key, value in self.options.items():
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
