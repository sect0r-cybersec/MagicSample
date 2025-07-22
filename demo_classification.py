#!/usr/bin/env python3.10
"""
Demo script for drum classification functionality
"""

import numpy as np
import librosa
import librosa.feature
import matplotlib.pyplot as plt

class DrumClassifierDemo:
    """Demo class for drum classification"""
    
    def __init__(self):
        self.categories = {
            'hihat': ['hihat', 'hi-hat', 'hi_hat', 'cymbal', 'crash', 'ride'],
            'snare': ['snare', 'clap'],
            'kick': ['kick', 'bass', 'bassdrum', 'bass_drum'],
            'tom': ['tom', 'floor_tom', 'rack_tom'],
            'clap': ['clap', 'hand_clap'],
            'percussion': ['perc', 'percussion', 'shaker', 'tambourine', 'cowbell']
        }
    
    def extract_features(self, audio_data, sample_rate):
        """Extract spectral features from audio"""
        features = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features['spectral_centroid'] = np.mean(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features['rms_energy'] = np.mean(rms)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zero_crossing_rate'] = np.mean(zcr)
        
        return features
    
    def classify_sample(self, audio_data, sample_rate):
        """Classify a drum sample based on its spectral characteristics"""
        try:
            features = self.extract_features(audio_data, sample_rate)
            
            # Classification logic
            if features['spectral_centroid'] > 4000 and features['zero_crossing_rate'] > 0.1:
                return 'hihat', features
            elif features['spectral_centroid'] < 1000 and features['rms_energy'] > 0.3:
                return 'kick', features
            elif 1000 < features['spectral_centroid'] < 3000 and features['spectral_bandwidth'] > 2000:
                return 'snare', features
            elif features['spectral_centroid'] < 2000 and features['spectral_rolloff'] < 3000:
                return 'tom', features
            elif features['zero_crossing_rate'] > 0.15:
                return 'clap', features
            else:
                return 'percussion', features
                
        except Exception as e:
            print(f"Classification error: {e}")
            return 'unknown', {}
    
    def create_synthetic_samples(self, sample_rate=44100, duration=0.1):
        """Create synthetic drum samples for demonstration"""
        samples = {}
        
        # Hi-hat (high frequency noise)
        t = np.linspace(0, duration, int(sample_rate * duration))
        hihat = np.random.normal(0, 0.3, len(t)) * np.exp(-t * 50)
        hihat = hihat * (1 + 0.5 * np.sin(2 * np.pi * 8000 * t))
        samples['hihat'] = hihat
        
        # Kick (low frequency thump)
        kick = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20)
        kick += 0.3 * np.random.normal(0, 1, len(t)) * np.exp(-t * 10)
        samples['kick'] = kick
        
        # Snare (mid frequency with noise)
        snare = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 15)
        snare += 0.5 * np.random.normal(0, 1, len(t)) * np.exp(-t * 8)
        samples['snare'] = snare
        
        # Tom (mid-low frequency)
        tom = np.sin(2 * np.pi * 120 * t) * np.exp(-t * 12)
        tom += 0.2 * np.random.normal(0, 1, len(t)) * np.exp(-t * 6)
        samples['tom'] = tom
        
        # Clap (high frequency transients)
        clap = np.random.normal(0, 0.4, len(t))
        clap *= np.exp(-t * 100)
        clap += 0.3 * np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 50)
        samples['clap'] = clap
        
        return samples
    
    def run_demo(self):
        """Run the classification demo"""
        print("Drum Classification Demo")
        print("=" * 40)
        
        # Create synthetic samples
        print("Creating synthetic drum samples...")
        samples = self.create_synthetic_samples()
        
        # Classify each sample
        results = {}
        for drum_type, audio_data in samples.items():
            print(f"\nAnalyzing {drum_type}...")
            predicted_type, features = self.classify_sample(audio_data, 44100)
            results[drum_type] = (predicted_type, features)
            
            print(f"  Actual: {drum_type}")
            print(f"  Predicted: {predicted_type}")
            print(f"  Features:")
            for feature_name, value in features.items():
                print(f"    {feature_name}: {value:.4f}")
        
        # Print summary
        print("\n" + "=" * 40)
        print("Classification Summary:")
        correct = 0
        total = len(results)
        
        for actual, (predicted, features) in results.items():
            status = "✓" if actual == predicted else "✗"
            print(f"{status} {actual:10} -> {predicted:10}")
            if actual == predicted:
                correct += 1
        
        accuracy = (correct / total) * 100
        print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{total})")
        
        # Feature visualization
        self.plot_features(results)
    
    def plot_features(self, results):
        """Plot feature distributions"""
        try:
            import matplotlib.pyplot as plt
            
            # Prepare data for plotting
            features_list = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'rms_energy', 'zero_crossing_rate']
            drum_types = list(results.keys())
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(features_list):
                values = [results[drum_type][1].get(feature, 0) for drum_type in drum_types]
                axes[i].bar(drum_types, values)
                axes[i].set_title(feature.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
            
            # Remove the last subplot if we have an odd number of features
            if len(features_list) < 6:
                axes[-1].remove()
            
            plt.tight_layout()
            plt.savefig('drum_classification_features.png', dpi=150, bbox_inches='tight')
            print("\nFeature visualization saved as 'drum_classification_features.png'")
            
        except ImportError:
            print("\nMatplotlib not available for visualization")

def main():
    """Main demo function"""
    demo = DrumClassifierDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 