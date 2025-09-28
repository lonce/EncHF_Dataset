#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
import tempfile

def calculate_windowed_rms(audio, sample_rate, window_ms=250):
    """Calculate RMS values over sliding windows"""
    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = window_samples // 4  # 75% overlap for better precision
    
    if len(audio) < window_samples:
        # If audio is shorter than window, return single RMS
        return np.array([np.sqrt(np.mean(audio**2))])
    
    rms_values = []
    for i in range(0, len(audio) - window_samples + 1, hop_samples):
        window = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    return np.array(rms_values)

def get_peak_windowed_rms(filepath, window_ms=250):
    """Get the peak RMS value from 250ms windows after initial processing"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Convert to mono and resample to 24kHz using sox
        cmd = [
            "sox", str(filepath), 
            "-r", "24000",  # sample rate
            "-c", "1",      # mono
            temp_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Load the processed audio
        audio, sr = sf.read(temp_path)
        
        # Calculate windowed RMS
        rms_values = calculate_windowed_rms(audio, sr, window_ms)
        peak_rms = np.max(rms_values)
        
        return peak_rms
        
    except subprocess.CalledProcessError as e:
        print(f"Sox error processing {filepath}: {e.stderr}")
        return None
    except Exception as e:
        print(f"Error calculating RMS for {filepath}: {e}")
        return None
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_file(input_path, output_path, target_peak_rms, window_ms=250):
    """Process a single audio file"""
    print(f"Processing: {input_path.name}")
    
    # Get current peak windowed RMS
    current_peak_rms = get_peak_windowed_rms(input_path, window_ms)
    
    if current_peak_rms is None or current_peak_rms == 0:
        print(f"  ✗ Skipping - could not calculate RMS")
        return False
    
    # Calculate gain needed
    gain_linear = target_peak_rms / current_peak_rms
    gain_db = 20 * np.log10(gain_linear)
    
    print(f"  Current peak RMS: {current_peak_rms:.6f}")
    print(f"  Target peak RMS: {target_peak_rms:.6f}")
    print(f"  Applying gain: {gain_db:.2f} dB")
    
    # Apply processing with sox
    cmd = [
        "sox", str(input_path),
        "-r", "24000",      # sample rate 24kHz
        "-c", "1",          # mono
        str(output_path),
        "gain", f"{gain_db:.2f}"  # apply calculated gain
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Saved: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Sox error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Normalize audio files to same peak 250ms windowed RMS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_normalize.py input_folder output_folder
  python audio_normalize.py input_folder output_folder --target-rms 0.05
  python audio_normalize.py ./wavs ./normalized --window-ms 500
        """
    )
    parser.add_argument('input_folder', help='Input folder containing wave files')
    parser.add_argument('output_folder', help='Output folder for processed files')
    parser.add_argument('--target-rms', type=float, default=0.1, 
                        help='Target peak windowed RMS value (default: 0.1)')
    parser.add_argument('--window-ms', type=int, default=250,
                        help='Window size in milliseconds (default: 250)')
    parser.add_argument('--extensions', nargs='+', default=['wav', 'WAV'],
                        help='Audio file extensions to process (default: wav WAV)')
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    if not input_folder.exists():
        print(f"Error: Input folder {input_folder} does not exist")
        return 1
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files with specified extensions
    audio_files = []
    for ext in args.extensions:
        audio_files.extend(input_folder.glob(f'*.{ext}'))
    
    if not audio_files:
        print(f"No audio files with extensions {args.extensions} found in {input_folder}")
        return 1
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Target peak windowed RMS: {args.target_rms}")
    print(f"Window size: {args.window_ms}ms")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print("=" * 60)
    
    success_count = 0
    
    for audio_file in sorted(audio_files):
        # Keep same filename but ensure .wav extension
        output_file = output_folder / f"{audio_file.stem}.wav"
        
        if process_file(audio_file, output_file, args.target_rms, args.window_ms):
            success_count += 1
        
        print()  # Empty line for readability
    
    print("=" * 60)
    print(f"Successfully processed {success_count}/{len(audio_files)} files")
    
    if success_count < len(audio_files):
        print(f"Failed to process {len(audio_files) - success_count} files")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
    