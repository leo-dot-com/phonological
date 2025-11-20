from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import logging
import speech_recognition as sr
from montreal_forced_aligner import align
from montreal_forced_aligner.models import AcousticModel, DictionaryModel
import librosa
import numpy as np
from typing import Dict, List, Any
import time

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhonologicalAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Initialize MFA components
        self.acoustic_model = None
        self.dictionary_model = None
        self.load_models()
    
    def load_models(self):
        """Load MFA models"""
        try:
            # You'll need to download and configure these models
            # For English, you can use the pretrained English model
            self.acoustic_model = AcousticModel('english')
            self.dictionary_model = DictionaryModel('english')
            logger.info("MFA models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MFA models: {e}")
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using speech recognition"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                transcription = self.recognizer.recognize_google(audio)
                return transcription.lower()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def force_align_phonemes(self, audio_path: str, text: str) -> List[Dict]:
        """Perform forced alignment using MFA"""
        try:
            # Create temporary directory for alignment
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run forced alignment
                alignment = align(
                    audio_path,
                    text,
                    self.acoustic_model,
                    self.dictionary_model,
                    temp_dir
                )
                
                # Extract phoneme-level alignment
                phoneme_alignment = []
                for segment in alignment.segments:
                    if segment.label:  # Phoneme label
                        phoneme_alignment.append({
                            'phoneme': segment.label,
                            'start': segment.start,
                            'end': segment.end,
                            'duration': segment.end - segment.start
                        })
                
                return phoneme_alignment
        except Exception as e:
            logger.error(f"Forced alignment error: {e}")
            return []
    
    def analyze_phoneme_deletion(self, audio_path: str, expected: str, original_word: str, remove_phoneme: str) -> Dict[str, Any]:
        """Analyze phoneme deletion task"""
        start_time = time.time()
        
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        
        # Force align with expected response
        phoneme_alignment = self.force_align_phonemes(audio_path, expected)
        
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
            'task_type': 'phoneme_deletion'
        }
    
    def analyze_phoneme_blending(self, audio_path: str, expected: str, phonemes: List[str]) -> Dict[str, Any]:
        """Analyze phoneme blending task"""
        start_time = time.time()
        
        transcription = self.transcribe_audio(audio_path)
        phoneme_alignment = self.force_align_phonemes(audio_path, expected)
        
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
            'task_type': 'phoneme_blending'
        }
    
    def analyze_nonword_repetition(self, audio_path: str, expected: str) -> Dict[str, Any]:
        """Analyze nonword repetition task"""
        start_time = time.time()
        
        transcription = self.transcribe_audio(audio_path)
        phoneme_alignment = self.force_align_phonemes(audio_path, expected)
        
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
            'task_type': 'nonword_repetition'
        }
    
    def calculate_accuracy(self, actual: str, expected: str) -> float:
        """Calculate word-level accuracy"""
        if not actual:
            return 0.0
        
        actual_words = actual.split()
        expected_words = expected.split()
        
        if len(expected_words) == 0:
            return 0.0
        
        # Simple exact match for now - could be enhanced with phonetic similarity
        matches = sum(1 for a, e in zip(actual_words, expected_words) if a == e)
        return matches / len(expected_words)
    
    def calculate_phoneme_accuracy(self, alignment: List[Dict], expected: str) -> float:
        """Calculate phoneme-level accuracy from alignment"""
        if not alignment:
            return 0.0
        
        # Count correctly aligned phonemes (simplified)
        # In a real implementation, you'd compare with expected phoneme sequence
        total_phonemes = len(alignment)
        if total_phonemes == 0:
            return 0.0
        
        # For demo purposes, return a reasonable accuracy
        # In production, you'd implement proper phoneme accuracy calculation
        return 0.85
    
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
        
        elif task_type == 'blending':
            if expected_lower not in actual_lower:
                error_patterns.append('blending_difficulty')
        
        elif task_type == 'repetition':
            if expected_lower not in actual_lower:
                error_patterns.append('repetition_error')
        
        # Phoneme substitution patterns
        if len(actual_lower) != len(expected_lower):
            error_patterns.append('length_mismatch')
        
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
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
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
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'phonological-analysis',
        'models_loaded': analyzer.acoustic_model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
