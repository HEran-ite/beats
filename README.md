# Beat HQ - AI Music Generation Platform

A powerful web-based music generation application that uses AI to create original musical compositions. Beat HQ supports both structured composition workflows and freeform text-to-music generation, making it suitable for both casual users and professional music creators.

![GitHub](https://img.shields.io/badge/GitHub-HEran--ite-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Flask](https://img.shields.io/badge/Flask-2.0+-red)

## 🎵 Features

### Dual Mode Interface

- **Basic Mode (Composition Mode)**: Create multi-section tracks with granular control over each section
  - Add unlimited sections (Intro, Middle, Bridge, Drop, Chorus, Ending)
  - Configure bars, chord progressions, instrumentation, and pitch curves
  - Drag-and-drop section reordering
  - Combine sections with automatic crossfading

- **Advanced Mode (Freeform Mode)**: Direct text-to-music generation
  - Natural language descriptions
  - Custom duration control (5-300 seconds)
  - Single-step complete track generation

### Core Capabilities

- **Multiple AI Models**: Choose from different MusicGen models trained on various datasets
- **Global Settings**: Set tempo (BPM), key, style, and preset for entire compositions
- **Real-Time Generation**: Live status updates and progress tracking
- **Audio Playback**: Integrated audio player with standard controls
- **Post-Processing**: 
  - Key and BPM correction
  - Spectral denoising
  - Harmonic enhancement
  - Professional mastering options
- **Authentication System**: Password-protected access with session management
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Supported Styles

- Trap, Hip Hop, Drill
- Electronic, EDM
- Rock, Alternative
- Jazz, Lo-Fi
- R&B, Pop
- Ambient, Cinematic
- Reggae

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for faster generation)
- 8GB+ RAM
- 10GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HEran-ite/beats.git
   cd beats
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchaudio
   pip install flask
   pip install audiocraft
   # Additional dependencies may be required - see audiocraft requirements
   ```

4. **Set up environment variables** (optional)
   ```bash
   export BEATS_PASSWORD="your_password_here"
   export FLASK_SECRET_KEY="your_secret_key_here"
   export BEATS_URL_PREFIX="/beats"  # Optional URL prefix
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   cd shared
   python app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:5001`
   - Default password: `beatworldhq` (or your custom `BEATS_PASSWORD`)

### Production Deployment

For production deployment, consider using:
- **Gunicorn** or **uWSGI** as WSGI server
- **Nginx** as reverse proxy (see `shared/nginx-beats-path.conf` for example config)
- **Systemd service** (see `shared/beats-app.service` for example)

## 📁 Project Structure

```
beats/
├── shared/                 # Main Flask application
│   ├── app.py             # Flask backend server
│   ├── musicgen_api.py    # MusicGen API integration
│   ├── templates/         # HTML templates
│   ├── static/           # CSS and JavaScript
│   └── utils/            # Utility functions
├── musicgen/              # Music generation modules
│   ├── audiocraft/       # AudioCraft integration
│   ├── postprocessing/   # Audio post-processing
│   ├── training/         # Model training scripts
│   └── test/             # Test scripts
├── musicgen_api.py        # Standalone MusicGen API script
├── FEATURES_DOCUMENTATION.md  # Detailed features documentation
├── COMPOSITION_COMBINATIONS.md # Composition examples and guides
└── README.md             # This file
```

## 🎹 Usage Examples

### Basic Mode - Creating a Full Track

1. Select **Basic Mode**
2. Set global settings:
   - Tempo: 140 BPM
   - Key: C minor
   - Style: Trap
   - Preset: Energetic
3. Add sections:
   - **Intro**: 4 bars, minimal instrumentation
   - **Middle**: 8 bars, full instrumentation
   - **Drop**: 8 bars, aggressive preset
   - **Chorus**: 8 bars, melodic preset
   - **Ending**: 4 bars, minimal instrumentation
4. Click **"CREATE BEAT"** to generate all sections
5. Preview individual sections
6. Click **"COMBINE SECTIONS"** to merge into a complete track

### Advanced Mode - Quick Generation

1. Select **Advanced Mode**
2. Enter a description:
   ```
   Dark trap beat with heavy 808s, fast hi-hats, and atmospheric pads. 
   Energetic and aggressive vibe at 140 BPM in C minor.
   ```
3. Set duration: 30 seconds
4. Click **"CREATE BEAT"**

## 📚 Documentation

- **[Features Documentation](FEATURES_DOCUMENTATION.md)**: Comprehensive guide to all features and workflows
- **[Composition Combinations](COMPOSITION_COMBINATIONS.md)**: Proven field combinations for different genres
- **[Post-Processing README](musicgen/postprocessing/README.md)**: Audio post-processing documentation

## 🔧 Configuration

### Environment Variables

- `BEATS_PASSWORD`: Authentication password (default: "beatworldhq")
- `FLASK_SECRET_KEY`: Secret key for Flask sessions
- `BEATS_URL_PREFIX`: URL prefix for reverse proxy setups (default: "/beats")
- `XFORMERS_DISABLED`: Set to "1" to disable xformers (recommended)

### Model Selection

The application supports multiple MusicGen models:
- `facebook/musicgen-small`: Fastest, lower quality
- `facebook/musicgen-medium`: Balanced (default)
- `facebook/musicgen-large`: Highest quality, slower
- `facebook/musicgen-stereo-melody-large`: Stereo with melody conditioning

## 🎨 Customization

### Styles and Presets

**Styles**: Trap, Hip Hop, Drill, Electronic, Rock, Jazz, Reggae, R&B, Pop, Ambient

**Presets**: Energetic, Chill, Aggressive, Melodic, Atmospheric, Minimal, Epic, Dark, Uplifting

### Instrumentation Options

Piano, Guitar, Bass, Drums, Synth, Strings, Brass, Pads, 808, Hi-Hats, Kick, Snare, Vocal Chops, Rhodes, Organ

## 🧪 Testing

Run the test suite:
```bash
cd musicgen/postprocessing
python test_postprocessing.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source. Please check individual component licenses (especially AudioCraft/MusicGen components).

## 🙏 Acknowledgments

- **Meta AI's AudioCraft**: For the MusicGen models and framework
- **Flask**: For the web framework
- All contributors and users of this project

## 📧 Contact

For questions, issues, or contributions:
- GitHub: [@HEran-ite](https://github.com/HEran-ite)
- Repository: [beats](https://github.com/HEran-ite/beats)

## 🔮 Future Enhancements

- [ ] Real-time collaboration features
- [ ] Additional AI models and styles
- [ ] MIDI export capabilities
- [ ] Cloud storage integration
- [ ] Advanced mixing and mastering tools
- [ ] Mobile app version

---

**Made with ❤️ for music creators**
