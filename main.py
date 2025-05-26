
import sys
import os
import threading
import numpy as np
import cv2
from PIL import Image
import librosa
import subprocess
import gc

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy import signal
    from scipy.io import wavfile

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QComboBox, QProgressBar, QFileDialog, QMessageBox,
                             QGroupBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor


class VideoGeneratorThread(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, audio_file, image_file, output_path, settings):
        super().__init__()
        self.audio_file = audio_file
        self.image_file = image_file
        self.output_path = output_path
        self.settings = settings

    def run(self):
        try:
            self.status_update.emit("Loading audio file...")
            self.progress_update.emit(10)

            # Load and analyze audio for visualization
            y, sr = librosa.load(self.audio_file, sr=None)
            duration = len(y) / sr

            # Apply audio effects if selected
            if self.settings.get('audio_effect', 'none') != 'none':
                self.status_update.emit("Applying audio effects...")
                self.progress_update.emit(15)
                y = self.apply_audio_effects(y, sr, self.settings['audio_effect'], self.settings['effect_intensity'])

                # Save processed audio to temporary file for final video
                temp_audio_path = self.audio_file.replace('.', '_processed.')
                if not temp_audio_path.endswith('.wav'):
                    temp_audio_path = temp_audio_path.rsplit('.', 1)[0] + '_processed.wav'

                if HAS_SOUNDFILE:
                    try:
                        sf.write(temp_audio_path, y, sr)
                        self.processed_audio_file = temp_audio_path
                    except:
                        # Fallback: use original audio if saving fails
                        self.processed_audio_file = self.audio_file
                else:
                    # Fallback: use scipy.io.wavfile to save
                    try:
                        if HAS_SCIPY:
                            # Convert to int16 for wav format
                            y_int16 = (y * 32767).astype(np.int16)
                            wavfile.write(temp_audio_path, sr, y_int16)
                            self.processed_audio_file = temp_audio_path
                        else:
                            # No scipy available, use original audio
                            self.processed_audio_file = self.audio_file
                    except:
                        # Final fallback: use original audio
                        self.processed_audio_file = self.audio_file
            else:
                self.processed_audio_file = self.audio_file

            self.status_update.emit("Loading image...")
            self.progress_update.emit(20)

            # Load and resize image
            resolution = self.settings['resolution'].split('x')
            width, height = int(resolution[0]), int(resolution[1])

            img = Image.open(self.image_file)
            img = img.resize((width, height), Image.Resampling.LANCZOS)

            self.status_update.emit("Preparing video creation...")
            self.progress_update.emit(30)

            # Generate video directly without storing all frames in memory
            fps = int(self.settings['fps'])
            total_frames = int(duration * fps)
            viz_type = self.settings['visualization']
            viz_position = self.settings['viz_position']

            # Create video from frames (without audio first)
            temp_video_path = self.output_path.replace('.mp4', '_temp.mp4')  # Use MP4 for better compatibility

            # Use better codec for temporary video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

            if not video_writer.isOpened():
                raise Exception("Could not open video writer. Try a different resolution or format.")

            self.status_update.emit("Creating video frames...")
            self.progress_update.emit(40)

            # Generate and write frames one by one to save memory
            img_array = np.array(img)
            samples_per_frame = len(y) // total_frames

            for frame_idx in range(total_frames):
                try:
                    # Get audio segment for this frame
                    start_sample = frame_idx * samples_per_frame
                    end_sample = min(start_sample + samples_per_frame, len(y))
                    audio_segment = y[start_sample:end_sample]

                    # Create frame with visualization
                    frame = self.create_single_frame(img_array.copy(), audio_segment, sr, viz_type, viz_position)

                    # Write frame to video
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    # Update progress
                    if frame_idx % 30 == 0:  # Update every 30 frames
                        progress = 40 + (frame_idx / total_frames) * 40
                        self.progress_update.emit(int(progress))

                    # Force garbage collection periodically
                    if frame_idx % 100 == 0:
                        gc.collect()

                except Exception as frame_error:
                    print(f"Error processing frame {frame_idx}: {frame_error}")
                    # Use previous frame or base image
                    frame = img_array.copy()
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            video_writer.release()

            self.status_update.emit("Adding audio...")
            self.progress_update.emit(85)

            # Use FFmpeg to combine video and audio
            success = self.combine_audio_video_ffmpeg(temp_video_path, self.processed_audio_file, self.output_path)

            # Clean up
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            # Clean up processed audio if it was created
            if hasattr(self, 'processed_audio_file') and self.processed_audio_file != self.audio_file:
                if os.path.exists(self.processed_audio_file):
                    os.remove(self.processed_audio_file)

            if success:
                self.status_update.emit("Video generated successfully!")
                self.progress_update.emit(100)
                self.finished_signal.emit(True, "Video generated successfully!")
            else:
                self.status_update.emit("Video created without audio")
                self.progress_update.emit(100)
                self.finished_signal.emit(False, "Video created but audio could not be synchronized.")

        except MemoryError:
            self.status_update.emit("Memory error - try lower resolution")
            self.finished_signal.emit(False,
                                      "Not enough memory to process this video.\nPlease try:\n1. Lower resolution (720p or 480p)\n2. Shorter audio file\n3. Close other applications")
        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False, f"An error occurred:\n{str(e)}")

    def create_single_frame(self, frame, audio_segment, sample_rate, viz_type, viz_position):
        """Create a single frame with visualization - memory efficient"""
        if len(audio_segment) > 0:
            try:
                if viz_type == "modern_waveform":
                    frame = self.add_modern_waveform(frame, audio_segment, viz_position)
                elif viz_type == "glow_bars":
                    frame = self.add_glow_bars(frame, audio_segment, viz_position)
                elif viz_type == "circular_bars":
                    frame = self.add_circular_bars(frame, audio_segment, viz_position)
                elif viz_type == "spectrum_wave":
                    frame = self.add_spectrum_wave(frame, audio_segment, sample_rate, viz_position)
                elif viz_type == "particle_wave":
                    frame = self.add_particle_wave(frame, audio_segment, viz_position)
                elif viz_type == "neon_bars":
                    frame = self.add_neon_bars(frame, audio_segment, viz_position)
            except Exception as e:
                print(f"Visualization error: {e}")
                pass
        return frame

    def add_modern_waveform(self, frame, audio_segment, position):
        """Modern waveform with gradient colors and glow effect"""
        height, width = frame.shape[:2]

        # Get visualization area
        viz_area = self.get_viz_area(frame, position)
        y_start, y_end, x_start, x_end = viz_area

        # Normalize audio
        if np.max(np.abs(audio_segment)) > 0:
            normalized_audio = audio_segment / np.max(np.abs(audio_segment))
        else:
            normalized_audio = audio_segment

        # Create smooth waveform with more points
        downsample_factor = max(1, len(normalized_audio) // 500)
        smooth_audio = normalized_audio[::downsample_factor]

        viz_height = y_end - y_start
        viz_width = x_end - x_start

        if position in ["left", "right"]:
            # Vertical waveformAudioViz Pro - Professional Audio Video Generator
            y_points = np.linspace(y_start, y_end - 1, len(smooth_audio)).astype(int)
            x_center = (x_start + x_end) // 2
            x_points = (smooth_audio * (viz_width // 3) + x_center).astype(int)
        else:
            # Horizontal waveform
            x_points = np.linspace(x_start, x_end - 1, len(smooth_audio)).astype(int)
            y_center = (y_start + y_end) // 2
            y_points = (smooth_audio * (viz_height // 3) + y_center).astype(int)

        # Draw waveform with gradient colors and glow
        for i in range(len(x_points) - 1):
            # Color gradient based on amplitude
            amplitude = abs(smooth_audio[i])
            hue = int(240 - 180 * amplitude)  # Blue to red gradient
            color = self.hsv_to_bgr(hue, 255, 255)

            # Draw multiple lines for glow effect
            for thickness in range(5, 0, -1):
                alpha = 0.3 + 0.7 * (6 - thickness) / 5  # More visible glow
                glow_color = tuple(int(c * alpha) for c in color)

                if position in ["left", "right"]:
                    cv2.line(frame, (x_points[i], y_points[i]),
                             (x_points[i + 1], y_points[i + 1]), glow_color, thickness)
                else:
                    cv2.line(frame, (x_points[i], y_points[i]),
                             (x_points[i + 1], y_points[i + 1]), glow_color, thickness)

        return frame

    def add_glow_bars(self, frame, audio_segment, position):
        """Glowing frequency bars with smooth animation"""
        height, width = frame.shape[:2]
        viz_area = self.get_viz_area(frame, position)
        y_start, y_end, x_start, x_end = viz_area

        num_bars = 64

        if len(audio_segment) > num_bars:
            # Compute FFT
            fft = np.fft.fft(audio_segment)
            fft_magnitude = np.abs(fft[:len(fft) // 2])

            # Group into bars with logarithmic spacing
            bar_heights = []
            for i in range(num_bars):
                start_idx = int((i / num_bars) ** 2 * len(fft_magnitude))
                end_idx = int(((i + 1) / num_bars) ** 2 * len(fft_magnitude))
                if end_idx > start_idx:
                    bar_height = np.mean(fft_magnitude[start_idx:end_idx])
                    bar_heights.append(bar_height)
                else:
                    bar_heights.append(0)

            # Normalize and smooth
            if np.max(bar_heights) > 0:
                bar_heights = np.array(bar_heights) / np.max(bar_heights)

            viz_height = y_end - y_start
            viz_width = x_end - x_start

            if position in ["top", "bottom", "center"]:
                bar_width = max(1, viz_width // num_bars)
                for i, height_norm in enumerate(bar_heights):
                    bar_height = max(2, int(height_norm * viz_height * 0.8))

                    x1 = x_start + i * bar_width + 1
                    x2 = min(x1 + bar_width - 2, x_end - 1)

                    if position == "bottom":
                        y1 = y_end
                        y2 = max(y_start, y_end - bar_height)
                    else:
                        y1 = y_start
                        y2 = min(y_end, y_start + bar_height)

                    # Create rainbow color based on frequency
                    hue = int(300 * i / num_bars)  # Purple to red
                    color = self.hsv_to_bgr(hue, 255, 255)

                    # Draw main bar
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

                    # Add bright highlight
                    highlight_color = self.hsv_to_bgr(hue, 200, 255)
                    cv2.rectangle(frame, (x1, y1), (x1 + 2, y2), highlight_color, -1)

        return frame

    def add_circular_bars(self, frame, audio_segment, position):
        """Circular frequency visualization"""
        height, width = frame.shape[:2]

        # Center the circle
        if position == "center":
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 6
        else:
            viz_area = self.get_viz_area(frame, position)
            y_start, y_end, x_start, x_end = viz_area
            center_x = (x_start + x_end) // 2
            center_y = (y_start + y_end) // 2
            radius = min(x_end - x_start, y_end - y_start) // 4

        num_bars = 72

        if len(audio_segment) > num_bars:
            # Compute FFT
            fft = np.fft.fft(audio_segment)
            fft_magnitude = np.abs(fft[:len(fft) // 2])

            # Group into bars
            bar_heights = []
            for i in range(num_bars):
                start_idx = i * len(fft_magnitude) // num_bars
                end_idx = (i + 1) * len(fft_magnitude) // num_bars
                bar_height = np.mean(fft_magnitude[start_idx:end_idx])
                bar_heights.append(bar_height)

            # Normalize
            if np.max(bar_heights) > 0:
                bar_heights = np.array(bar_heights) / np.max(bar_heights)

            # Draw circular bars
            for i, height_norm in enumerate(bar_heights):
                angle = 2 * np.pi * i / num_bars
                bar_length = max(5, int(height_norm * radius * 0.8))

                # Inner and outer points
                inner_x = int(center_x + radius * np.cos(angle))
                inner_y = int(center_y + radius * np.sin(angle))
                outer_x = int(center_x + (radius + bar_length) * np.cos(angle))
                outer_y = int(center_y + (radius + bar_length) * np.sin(angle))

                # Color based on angle
                hue = int(360 * i / num_bars)
                color = self.hsv_to_bgr(hue, 255, 255)

                # Draw line with thickness based on amplitude
                thickness = max(2, int(height_norm * 6))
                cv2.line(frame, (inner_x, inner_y), (outer_x, outer_y), color, thickness)

        return frame

    def add_spectrum_wave(self, frame, audio_segment, sample_rate, position):
        """Spectrum analyzer with wave-like motion"""
        height, width = frame.shape[:2]
        viz_area = self.get_viz_area(frame, position)
        y_start, y_end, x_start, x_end = viz_area

        if len(audio_segment) > 1024:
            # Compute spectrogram
            freqs, times, Sxx = librosa.stft(audio_segment, n_fft=1024, hop_length=512)
            Sxx_db = librosa.amplitude_to_db(np.abs(Sxx))

            viz_height = y_end - y_start
            viz_width = x_end - x_start

            # Create wave pattern
            for x in range(x_start, x_end, 3):
                rel_x = (x - x_start) / viz_width
                freq_idx = int(rel_x * Sxx_db.shape[0])

                if freq_idx < Sxx_db.shape[0]:
                    # Get amplitude at this frequency
                    amplitude = np.mean(Sxx_db[freq_idx, :])
                    normalized_amp = (amplitude + 80) / 80  # Normalize dB values
                    normalized_amp = max(0, min(1, normalized_amp))

                    # Create wave motion
                    wave_height = int(normalized_amp * viz_height * 0.6)
                    wave_center = (y_start + y_end) // 2

                    # Draw vertical wave line
                    y1 = max(y_start, wave_center - wave_height // 2)
                    y2 = min(y_end, wave_center + wave_height // 2)

                    # Color based on frequency
                    hue = int(240 * (1 - rel_x))  # Blue to red
                    color = self.hsv_to_bgr(hue, 255, 255)

                    cv2.line(frame, (x, y1), (x, y2), color, 2)

        return frame

    def add_particle_wave(self, frame, audio_segment, position):
        """Particle-based waveform visualization"""
        height, width = frame.shape[:2]
        viz_area = self.get_viz_area(frame, position)
        y_start, y_end, x_start, x_end = viz_area

        # Normalize audio
        if np.max(np.abs(audio_segment)) > 0:
            normalized_audio = audio_segment / np.max(np.abs(audio_segment))
        else:
            normalized_audio = audio_segment

        # Downsample for particles
        num_particles = min(100, x_end - x_start)
        downsample_factor = max(1, len(normalized_audio) // num_particles)
        particles = normalized_audio[::downsample_factor]

        viz_height = y_end - y_start
        viz_width = x_end - x_start

        # Draw particles
        for i, amplitude in enumerate(particles):
            if i >= num_particles:
                break

            x = x_start + int(i * viz_width / len(particles))
            y_center = (y_start + y_end) // 2
            y = int(y_center + amplitude * viz_height * 0.3)
            y = max(y_start, min(y_end, y))  # Clamp to bounds

            # Particle size based on amplitude
            radius = max(2, int(abs(amplitude) * 8) + 2)

            # Color based on amplitude
            if amplitude > 0:
                color = self.hsv_to_bgr(60, 255, 255)  # Yellow for positive
            else:
                color = self.hsv_to_bgr(240, 255, 255)  # Blue for negative

            # Draw particle
            cv2.circle(frame, (x, y), radius, color, -1)

            # Add smaller bright center
            cv2.circle(frame, (x, y), max(1, radius // 2), (255, 255, 255), -1)

        return frame

    def add_neon_bars(self, frame, audio_segment, position):
        """Neon-style bars with electric effect"""
        height, width = frame.shape[:2]
        viz_area = self.get_viz_area(frame, position)
        y_start, y_end, x_start, x_end = viz_area

        num_bars = 32

        if len(audio_segment) > num_bars:
            # Compute FFT
            fft = np.fft.fft(audio_segment)
            fft_magnitude = np.abs(fft[:len(fft) // 2])

            # Group into bars
            bar_heights = []
            for i in range(num_bars):
                start_idx = i * len(fft_magnitude) // num_bars
                end_idx = (i + 1) * len(fft_magnitude) // num_bars
                bar_height = np.mean(fft_magnitude[start_idx:end_idx])
                bar_heights.append(bar_height)

            # Normalize
            if np.max(bar_heights) > 0:
                bar_heights = np.array(bar_heights) / np.max(bar_heights)

            viz_height = y_end - y_start
            viz_width = x_end - x_start
            bar_width = max(1, viz_width // num_bars)

            for i, height_norm in enumerate(bar_heights):
                bar_height = max(2, int(height_norm * viz_height * 0.9))

                x1 = x_start + i * bar_width + 2
                x2 = min(x1 + bar_width - 4, x_end - 1)

                if position == "bottom":
                    y1 = y_end
                    y2 = max(y_start, y_end - bar_height)
                else:
                    y1 = y_start
                    y2 = min(y_end, y_start + bar_height)

                # Neon colors (cyan, magenta, yellow)
                if i % 3 == 0:
                    color = self.hsv_to_bgr(180, 255, 255)  # Cyan
                elif i % 3 == 1:
                    color = self.hsv_to_bgr(300, 255, 255)  # Magenta
                else:
                    color = self.hsv_to_bgr(60, 255, 255)  # Yellow

                # Draw main bar
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

                # Add bright white core
                core_width = max(1, (x2 - x1) // 3)
                core_x = x1 + (x2 - x1) // 2 - core_width // 2
                cv2.rectangle(frame, (core_x, y1), (core_x + core_width, y2), (255, 255, 255), -1)

        return frame

    def get_viz_area(self, frame, position):
        """Get visualization area coordinates"""
        height, width = frame.shape[:2]

        if position == "top":
            return 0, height // 6, 0, width
        elif position == "bottom":
            return height - height // 6, height, 0, width
        elif position == "left":
            return 0, height, 0, width // 6
        elif position == "right":
            return 0, height, width - width // 6, width
        elif position == "center":
            viz_height = height // 4
            y_start = height // 2 - viz_height // 2
            return y_start, y_start + viz_height, 0, width
        else:  # default to top
            return 0, height // 6, 0, width

    def hsv_to_bgr(self, h, s, v):
        """Convert HSV to BGR color for OpenCV"""
        h = h % 360
        s = s / 255.0
        v = v / 255.0

        c = v * s
        x = c * (1 - abs((h / 60.0) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        # Convert to BGR and scale to 0-255
        return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))

    def combine_audio_video_ffmpeg(self, video_path, audio_path, output_path):
        """Combine video and audio using FFmpeg with social media optimization"""
        try:
            # Get quality and social media settings
            quality = self.settings.get('quality', 'high')
            social_preset = self.settings.get('social_preset', 'custom')

            # Base FFmpeg command
            cmd = ['ffmpeg', '-y']  # -y to overwrite output file

            # Input files
            cmd.extend(['-i', video_path])  # input video
            cmd.extend(['-i', audio_path])  # input audio

            # Video codec settings optimized for social media
            if quality == 'fast':
                # Fast encoding for quick previews
                cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28'])
            elif quality == 'medium':
                # Balanced quality and speed
                cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '23'])
            else:  # high quality
                # Best quality for social media
                cmd.extend(['-c:v', 'libx264', '-preset', 'medium', '-crf', '20'])

            # Audio codec settings
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])

            # Social media specific optimizations
            if social_preset in ['instagram_post', 'instagram_story', 'tiktok', 'youtube_shorts']:
                # Mobile-optimized settings
                cmd.extend(['-profile:v', 'high', '-level', '4.0'])
                cmd.extend(['-pix_fmt', 'yuv420p'])  # Ensure compatibility
                cmd.extend(['-movflags', '+faststart'])  # Enable fast streaming

                if social_preset == 'instagram_post':
                    # Instagram square video optimization
                    cmd.extend(['-b:v', '3500k'])  # Good bitrate for square videos
                elif social_preset in ['instagram_story', 'tiktok', 'youtube_shorts']:
                    # Vertical video optimization
                    cmd.extend(['-b:v', '5000k'])  # Higher bitrate for vertical videos

            elif social_preset in ['facebook', 'twitter']:
                # Web platform optimization
                cmd.extend(['-profile:v', 'high', '-level', '4.0'])
                cmd.extend(['-pix_fmt', 'yuv420p'])
                cmd.extend(['-movflags', '+faststart'])
                cmd.extend(['-b:v', '4000k'])  # Good web streaming bitrate

            else:  # custom
                # General social media optimization
                cmd.extend(['-profile:v', 'high', '-level', '4.0'])
                cmd.extend(['-pix_fmt', 'yuv420p'])
                cmd.extend(['-movflags', '+faststart'])

            # Additional optimizations for all social media
            cmd.extend(['-shortest'])  # Finish when shortest input ends
            cmd.extend(['-threads', '0'])  # Use all available CPU cores

            # Output file
            cmd.append(output_path)

            # Run FFmpeg
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Verify output file exists and has reasonable size
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    return True
                else:
                    print(f"Output file verification failed")
                    return False
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False

        except FileNotFoundError:
            print("FFmpeg not found")
            return False
        except Exception as e:
            print(f"Error in combine_audio_video_ffmpeg: {e}")
            return False

    def apply_audio_effects(self, audio, sample_rate, effect_type, intensity):
        """Apply radio channel effects to audio"""
        try:
            # Convert intensity to numerical values
            intensity_values = {"light": 0.3, "medium": 0.6, "heavy": 0.9}
            intensity_factor = intensity_values.get(intensity, 0.6)

            if effect_type == "fm_radio":
                return self.apply_fm_radio_effect(audio, sample_rate, intensity_factor)
            elif effect_type == "am_radio":
                return self.apply_am_radio_effect(audio, sample_rate, intensity_factor)
            elif effect_type == "vintage_fm":
                return self.apply_vintage_fm_effect(audio, sample_rate, intensity_factor)
            elif effect_type == "lo_fi_am":
                return self.apply_lo_fi_am_effect(audio, sample_rate, intensity_factor)
            else:
                return audio
        except Exception as e:
            print(f"Error applying audio effects: {e}")
            return audio

    def apply_fm_radio_effect(self, audio, sr, intensity):
        """Apply FM radio channel effect"""
        # FM radio characteristics: wider frequency response, less noise
        processed = audio.copy()

        # 1. Add subtle compression (FM stations compress audio)
        processed = self.apply_compression(processed, ratio=2.0 * intensity)

        # 2. Apply FM frequency response (boost highs slightly)
        processed = self.apply_eq_curve(processed, sr, 'fm', intensity)

        # 3. Add very subtle static/noise
        noise_level = 0.002 * intensity
        noise = np.random.normal(0, noise_level, len(processed))
        processed = processed + noise

        # 4. Apply slight stereo widening effect (if stereo)
        if len(processed.shape) > 1:
            processed = self.apply_stereo_widening(processed, intensity * 0.3)

        # 5. Add subtle tape saturation
        processed = self.apply_tape_saturation(processed, intensity * 0.4)

        return np.clip(processed, -1.0, 1.0)

    def apply_am_radio_effect(self, audio, sr, intensity):
        """Apply AM radio channel effect"""
        # AM radio characteristics: limited frequency response, more noise
        processed = audio.copy()

        # 1. Apply AM frequency response (limited bandwidth 300Hz-3kHz)
        processed = self.apply_bandpass_filter(processed, sr, 300, 3000, intensity)

        # 2. Add compression (AM has heavy compression)
        processed = self.apply_compression(processed, ratio=4.0 * intensity)

        # 3. Add static noise
        noise_level = 0.01 * intensity
        noise = np.random.normal(0, noise_level, len(processed))
        processed = processed + noise

        # 4. Apply amplitude modulation artifacts
        processed = self.apply_am_artifacts(processed, sr, intensity)

        # 5. Add slight distortion
        processed = self.apply_soft_clipping(processed, intensity * 0.6)

        return np.clip(processed, -1.0, 1.0)

    def apply_vintage_fm_effect(self, audio, sr, intensity):
        """Apply vintage FM radio effect"""
        processed = audio.copy()

        # Vintage FM has warmer sound, slight wow and flutter
        processed = self.apply_fm_radio_effect(processed, sr, intensity * 0.8)

        # Add wow and flutter (tape-like modulation)
        processed = self.apply_wow_flutter(processed, sr, intensity * 0.5)

        # Add vintage warmth (slight low-pass filtering)
        processed = self.apply_vintage_warmth(processed, sr, intensity)

        return np.clip(processed, -1.0, 1.0)

    def apply_lo_fi_am_effect(self, audio, sr, intensity):
        """Apply lo-fi AM radio effect"""
        processed = audio.copy()

        # Start with AM effect
        processed = self.apply_am_radio_effect(processed, sr, intensity)

        # Add lo-fi characteristics
        # 1. Reduce bit depth (bit crushing)
        processed = self.apply_bit_crushing(processed, intensity)

        # 2. Add vinyl crackle
        processed = self.add_vinyl_crackle(processed, sr, intensity)

        # 3. More aggressive filtering
        processed = self.apply_bandpass_filter(processed, sr, 400, 2500, intensity)

        return np.clip(processed, -1.0, 1.0)

    def apply_compression(self, audio, ratio=2.0, threshold=0.5):
        """Apply audio compression"""
        compressed = audio.copy()

        # Simple compression algorithm
        above_threshold = np.abs(compressed) > threshold
        compressed[above_threshold] = np.sign(compressed[above_threshold]) * (
                threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio
        )

        return compressed

    def apply_eq_curve(self, audio, sr, curve_type, intensity):
        """Apply EQ curves for different radio types"""
        if not HAS_SCIPY:
            return audio

        try:
            # Create different EQ curves
            if curve_type == 'fm':
                # FM: slight high-frequency boost
                b, a = signal.butter(2, 8000 / (sr / 2), btype='highpass')
                eq_signal = signal.filtfilt(b, a, audio)
                return audio + intensity * 0.3 * eq_signal
        except:
            pass

        return audio

    def apply_bandpass_filter(self, audio, sr, low_freq, high_freq, intensity):
        """Apply bandpass filter"""
        if not HAS_SCIPY:
            return audio

        try:
            # Design bandpass filter
            nyquist = sr / 2
            low = low_freq / nyquist
            high = min(high_freq / nyquist, 0.99)  # Ensure < 1

            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, audio)

            # Mix with original based on intensity
            return audio * (1 - intensity) + filtered * intensity
        except:
            pass

        return audio

    def apply_tape_saturation(self, audio, intensity):
        """Apply tape saturation effect"""
        # Soft saturation using tanh
        saturated = np.tanh(audio * (1 + intensity * 2))
        return audio * (1 - intensity) + saturated * intensity

    def apply_soft_clipping(self, audio, intensity):
        """Apply soft clipping distortion"""
        clipped = np.tanh(audio * (1 + intensity * 3))
        return audio * (1 - intensity) + clipped * intensity

    def apply_am_artifacts(self, audio, sr, intensity):
        """Apply AM radio artifacts"""
        # Add slight amplitude modulation
        t = np.arange(len(audio)) / sr

        # Low-frequency amplitude modulation (simulates signal fading)
        am_freq = 0.5 + np.random.random() * 2  # 0.5-2.5 Hz
        am_mod = 1 + intensity * 0.1 * np.sin(2 * np.pi * am_freq * t)

        return audio * am_mod

    def apply_wow_flutter(self, audio, sr, intensity):
        """Apply wow and flutter effect"""
        # Wow and flutter: slow and fast pitch variations
        t = np.arange(len(audio)) / sr

        # Wow (slow variations, <10Hz)
        wow_freq = 0.5 + np.random.random() * 2
        wow_mod = intensity * 0.002 * np.sin(2 * np.pi * wow_freq * t)

        # Flutter (fast variations, 10-100Hz)
        flutter_freq = 10 + np.random.random() * 20
        flutter_mod = intensity * 0.001 * np.sin(2 * np.pi * flutter_freq * t)

        # Apply modulation (simplified - just amplitude modulation)
        total_mod = 1 + wow_mod + flutter_mod
        return audio * total_mod

    def apply_vintage_warmth(self, audio, sr, intensity):
        """Apply vintage warmth (low-pass filtering)"""
        if not HAS_SCIPY:
            return audio

        try:
            # Gentle low-pass filter for warmth
            cutoff = 8000 - intensity * 2000  # 6kHz to 8kHz
            b, a = signal.butter(2, cutoff / (sr / 2), btype='lowpass')
            warm = signal.filtfilt(b, a, audio)

            return audio * (1 - intensity * 0.3) + warm * (intensity * 0.3)
        except:
            pass

        return audio

    def apply_bit_crushing(self, audio, intensity):
        """Apply bit crushing for lo-fi effect"""
        # Reduce effective bit depth
        bit_depth = 16 - int(intensity * 8)  # 16 to 8 bits
        levels = 2 ** bit_depth

        crushed = np.round(audio * levels) / levels
        return audio * (1 - intensity) + crushed * intensity

    def add_vinyl_crackle(self, audio, sr, intensity):
        """Add vinyl crackle noise"""
        # Generate crackle using filtered noise
        crackle_length = len(audio)

        # High-frequency noise bursts
        crackle = np.random.normal(0, 0.001, crackle_length)

        # Make it sparse (occasional pops)
        pop_probability = intensity * 0.0001
        pop_mask = np.random.random(crackle_length) < pop_probability
        crackle = crackle * pop_mask

        # Add some high-frequency emphasis to crackle if scipy available
        if HAS_SCIPY:
            try:
                b, a = signal.butter(2, 4000 / (sr / 2), btype='highpass')
                crackle = signal.filtfilt(b, a, crackle)
            except:
                pass

        return audio + crackle * intensity

    def apply_stereo_widening(self, audio, intensity):
        """Apply stereo widening effect"""
        if len(audio.shape) < 2:
            return audio  # Mono audio

        # Simple stereo widening
        left = audio[:, 0]
        right = audio[:, 1]

        # Create wider stereo image
        mid = (left + right) / 2
        side = (left - right) / 2

        # Enhance stereo width
        wide_left = mid + side * (1 + intensity)
        wide_right = mid - side * (1 + intensity)

        return np.column_stack([wide_left, wide_right])


class AudioVideoGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_file = None
        self.image_file = None
        self.output_path = None
        self.worker_thread = None

        self.init_ui()
        self.set_style()

    def init_ui(self):
        self.setWindowTitle("AudioViz Pro - Professional Audio Video Generator (Ali Hayajneh)")
        self.setGeometry(100, 100, 900, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title_label = QLabel("AudioViz Pro - Professional Audio Video Generator (Ali Hayajneh)")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 18, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout(file_group)

        # Audio file
        file_layout.addWidget(QLabel("Audio File (.wav/.mp3):"), 0, 0)
        self.audio_label = QLabel("No file selected")
        self.audio_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.audio_label, 0, 1)
        audio_btn = QPushButton("Browse")
        audio_btn.clicked.connect(self.select_audio_file)
        file_layout.addWidget(audio_btn, 0, 2)

        # Image file
        file_layout.addWidget(QLabel("Image File:"), 1, 0)
        self.image_label = QLabel("No file selected")
        self.image_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.image_label, 1, 1)
        image_btn = QPushButton("Browse")
        image_btn.clicked.connect(self.select_image_file)
        file_layout.addWidget(image_btn, 1, 2)

        # Output file
        file_layout.addWidget(QLabel("Output Video:"), 2, 0)
        self.output_label = QLabel("No path selected")
        self.output_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.output_label, 2, 1)
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.select_output_path)
        file_layout.addWidget(output_btn, 2, 2)

        file_layout.setColumnStretch(1, 1)
        main_layout.addWidget(file_group)

        # Options group
        options_group = QGroupBox("Video Options")
        options_layout = QGridLayout(options_group)

        # Resolution options - updated with social media formats
        options_layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x360", "854x480", "1280x720", "1080x1080", "1080x1920", "1920x1080"])
        self.resolution_combo.setCurrentText("1280x720")
        options_layout.addWidget(self.resolution_combo, 0, 1)

        # FPS
        options_layout.addWidget(QLabel("FPS:"), 0, 2)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "30", "60"])
        self.fps_combo.setCurrentText("30")
        options_layout.addWidget(self.fps_combo, 0, 3)

        # Visualization type
        options_layout.addWidget(QLabel("Visualization:"), 1, 0)
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(
            ["modern_waveform", "glow_bars", "circular_bars", "spectrum_wave", "particle_wave", "neon_bars"])
        options_layout.addWidget(self.viz_combo, 1, 1)

        # Visualization position
        options_layout.addWidget(QLabel("Position:"), 1, 2)
        self.position_combo = QComboBox()
        self.position_combo.addItems(["top", "bottom", "left", "right", "center"])
        options_layout.addWidget(self.position_combo, 1, 3)

        # Audio effects
        options_layout.addWidget(QLabel("Audio Effect:"), 2, 0)
        self.audio_effect_combo = QComboBox()
        self.audio_effect_combo.addItems(["none", "fm_radio", "am_radio", "vintage_fm", "lo_fi_am"])
        options_layout.addWidget(self.audio_effect_combo, 2, 1)

        # Effect intensity
        options_layout.addWidget(QLabel("Effect Intensity:"), 2, 2)
        self.intensity_combo = QComboBox()
        self.intensity_combo.addItems(["light", "medium", "heavy"])
        self.intensity_combo.setCurrentText("medium")
        options_layout.addWidget(self.intensity_combo, 2, 3)

        # Social media preset
        options_layout.addWidget(QLabel("Social Media:"), 3, 0)
        self.social_preset_combo = QComboBox()
        self.social_preset_combo.addItems(
            ["custom", "instagram_post", "instagram_story", "tiktok", "youtube_shorts", "facebook", "twitter"])
        self.social_preset_combo.currentTextChanged.connect(self.apply_social_preset)
        options_layout.addWidget(self.social_preset_combo, 3, 1)

        # Video quality
        options_layout.addWidget(QLabel("Quality:"), 3, 2)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["high", "medium", "fast"])
        self.quality_combo.setCurrentText("high")
        options_layout.addWidget(self.quality_combo, 3, 3)

        main_layout.addWidget(options_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to generate video")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)

        main_layout.addWidget(progress_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Video")
        self.generate_btn.setFixedHeight(50)
        self.generate_btn.clicked.connect(self.start_generation)
        main_layout.addWidget(self.generate_btn)

        # Add stretch
        main_layout.addStretch()

    def set_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)

    def select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio files (*.wav *.mp3);;All files (*.*)"
        )
        if file_path:
            self.audio_file = file_path
            self.audio_label.setText(os.path.basename(file_path))
            self.audio_label.setStyleSheet("color: black;")

    def select_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*.*)"
        )
        if file_path:
            self.image_file = file_path
            self.image_label.setText(os.path.basename(file_path))
            self.image_label.setStyleSheet("color: black;")

    def select_output_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video As", "",
            "MP4 files (*.mp4);;All files (*.*)"
        )
        if file_path:
            self.output_path = file_path
            self.output_label.setText(os.path.basename(file_path))
            self.output_label.setStyleSheet("color: black;")

    def start_generation(self):
        if not self.validate_inputs():
            return

        # Disable generate button
        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # Prepare settings
        settings = {
            'resolution': self.resolution_combo.currentText(),
            'fps': self.fps_combo.currentText(),
            'visualization': self.viz_combo.currentText(),
            'viz_position': self.position_combo.currentText(),
            'audio_effect': self.audio_effect_combo.currentText(),
            'effect_intensity': self.intensity_combo.currentText(),
            'social_preset': self.social_preset_combo.currentText(),
            'quality': self.quality_combo.currentText()
        }

        # Start worker thread
        self.worker_thread = VideoGeneratorThread(
            self.audio_file, self.image_file, self.output_path, settings
        )
        self.worker_thread.progress_update.connect(self.progress_bar.setValue)
        self.worker_thread.status_update.connect(self.status_label.setText)
        self.worker_thread.finished_signal.connect(self.on_generation_finished)
        self.worker_thread.start()

    def validate_inputs(self):
        if not self.audio_file:
            QMessageBox.critical(self, "Error", "Please select an audio file")
            return False
        if not self.image_file:
            QMessageBox.critical(self, "Error", "Please select an image file")
            return False
        if not self.output_path:
            QMessageBox.critical(self, "Error", "Please select output path")
            return False
        return True

    def on_generation_finished(self, success, message):
        self.generate_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", f"Video saved to:\n{self.output_path}")
        else:
            if "audio could not be synchronized" in message:
                QMessageBox.warning(self, "Audio Not Added", message)
            else:
                QMessageBox.critical(self, "Error", message)

    def apply_social_preset(self, preset):
        """Apply social media platform presets"""
        presets = {
            "instagram_post": {"resolution": "1080x1080", "fps": "30"},  # Square format
            "instagram_story": {"resolution": "1080x1920", "fps": "30"},  # Vertical 9:16
            "tiktok": {"resolution": "1080x1920", "fps": "30"},  # Vertical 9:16
            "youtube_shorts": {"resolution": "1080x1920", "fps": "30"},  # Vertical 9:16
            "facebook": {"resolution": "1280x720", "fps": "30"},  # Standard HD
            "twitter": {"resolution": "1280x720", "fps": "30"},  # Standard HD
            "custom": {}  # No changes
        }

        if preset in presets and presets[preset]:
            settings = presets[preset]
            if "resolution" in settings:
                # Update resolution combo if the resolution exists in the list
                index = self.resolution_combo.findText(settings["resolution"])
                if index >= 0:
                    self.resolution_combo.setCurrentIndex(index)
                else:
                    # Add custom resolution if not in list
                    self.resolution_combo.addItem(settings["resolution"])
                    self.resolution_combo.setCurrentText(settings["resolution"])

            if "fps" in settings:
                self.fps_combo.setCurrentText(settings["fps"])


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = AudioVideoGenerator()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()