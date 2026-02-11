# Transight - Intelligent Transit Prediction System

## Project Overview

**Transight** is a real-time bus arrival prediction system for Route 72 in Bristol, UK (Temple Meads to UWE Frenchay). It combines live bus GPS tracking, traffic data, and computer vision to provide accurate arrival time estimates.

### Key Features

1. **Real-Time Bus Tracking**: Shows live bus locations on a map, updating every 10 seconds from BODS (Bus Open Data Service)
2. **Traffic-Aware Predictions**: Uses TomTom Traffic Flow API for real-time road conditions and adjusts ETAs during congestion
3. **Crowd Detection (Computer Vision)**: YOLOv8 AI model counts people at bus stops and factors boarding time into predictions
4. **Machine Learning Predictions**: XGBoost model trained on historical data predicts arrival delays based on traffic, crowd, and weather
5. **Interactive Web Interface**: Leaflet map showing Route 72 line and stops with live ETA breakdown

---

## Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | FastAPI (Python 3.12) | REST API server |
| ML Model | XGBoost | Arrival delay prediction |
| Data Processing | pandas, numpy | Data manipulation |
| Computer Vision | Ultralytics YOLOv8, OpenCV | People counting at stops |
| HTTP Client | requests | External API calls |
| ASGI Server | uvicorn | Production server |

### Frontend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | React 19 | UI components |
| Build Tool | Vite 7 | Development & bundling |
| UI Library | Material-UI (MUI) v6 | Buttons, cards, icons |
| Maps | Leaflet + React-Leaflet | Interactive map display |
| HTTP Client | Axios | API requests |

### External APIs

- **BODS API**: UK Department for Transport - Real-time bus GPS locations (SIRI-VM XML format)
- **TomTom Traffic API**: Road speed & congestion data (Free tier: 2,500 requests/day)

---

## Project Structure

```
Bus Predictor/
├── server.py                   # FastAPI backend server (entry point)
├── pedestrian_detector.py      # YOLOv8 computer vision module
├── model_trainer.py            # XGBoost ML training pipeline
├── data_generator.py           # Synthetic training data generator
├── camera_launcher.py          # Multi-camera orchestration launcher
├── system_config.json          # System configuration
├── trained_model.json          # Trained XGBoost model (generated)
├── yolo.pt                     # YOLOv8 weights file
├── historical_bus_data.csv     # Training dataset (generated)
├── README.md                   # User documentation
├── .venv/                      # Python virtual environment
├── frontend/                   # React frontend application
│   ├── package.json            # Node.js dependencies
│   ├── vite.config.js          # Vite configuration
│   ├── eslint.config.js        # ESLint configuration
│   ├── index.html              # HTML entry point
│   └── src/
│       ├── TransitApp.jsx      # Main React application component
│       ├── main.jsx            # React entry point
│       ├── index.css           # Global styles
│       └── assets/             # Static assets
├── videos/                     # CV input videos
│   ├── crowd_quiet.mp4
│   └── crowd_busy.mp4
└── runs/                       # YOLO detection output directory
```

---

## Build and Run Commands

### Backend Setup

```bash
cd "Bus Predictor"

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install Python dependencies
pip install fastapi uvicorn requests xgboost pandas ultralytics opencv-python scikit-learn

# Generate training data (if historical_bus_data.csv doesn't exist)
python data_generator.py

# Train ML model (if trained_model.json doesn't exist)
python model_trainer.py

# Start the backend server
python server.py
```

The backend server will start on `http://localhost:8000` with API documentation at `/docs`.

**API Keys**: The system is pre-configured with working API keys:
- BODS API Key: For real-time bus GPS data
- TomTom API Key: For real-time traffic congestion data

### Frontend Setup

```bash
cd "Bus Predictor/frontend"

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### Running All Three Data Sources Together

For the most accurate predictions, you need to run all three data collection components:

#### Terminal 1: Start Backend Server
```bash
cd "Bus Predictor"
.venv\Scripts\activate
python server.py
```

#### Terminal 2: Start Frontend
```bash
cd "Bus Predictor/frontend"
npm install  # First time only
npm run dev
```

#### Terminal 3: Start CV Module (for passenger counting)
```bash
cd "Bus Predictor"
.venv\Scripts\activate

# For Temple Meads station
python pedestrian_detector.py --station STATION_TM01 --video crowd_busy.mp4

# For other stations, use their station IDs:
# STATION_TM01 = Temple Meads
# STATION_CC02 = The Centre
# STATION_CB03 = Cabot Circus
# etc.
```

The CV module will:
1. Process video using YOLOv8 to detect people
2. Count passengers in the waiting zone
3. Send counts to the backend every 5 seconds
4. Display the detection visualization

### Computer Vision Module (Optional)

```bash
cd "Bus Predictor"

# Single camera mode
python pedestrian_detector.py --station STATION_TM01 --video crowd_busy.mp4

# Multi-camera mode (runs multiple detectors)
python camera_launcher.py
```

---

## API Endpoints

### Bus Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stations` | GET | List all transit stations |
| `/api/stations/nearby` | GET | Find stations near coordinates |
| `/api/stations/{station_id}/approaching` | GET | Get buses approaching a specific stop with ETAs |
| `/api/vehicles/active` | GET | Get all currently tracked vehicles |
| `/api/route/geometry` | GET | Get route path coordinates |

### Sensor Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sensors/update` | POST | Receive crowd count from CV sensor |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System status check (includes API configuration status) |
| `/` | GET | API information |
| `/docs` | GET | Swagger UI documentation |

---

## Code Style Guidelines

### Python Code Style

- **Docstrings**: Use triple-quoted docstrings for module and function documentation
- **Type Hints**: Use type annotations for function parameters and return values
- **Constants**: Define configuration constants at module level in UPPER_CASE
- **Naming**: Use `snake_case` for variables and functions, `PascalCase` for classes
- **Comments**: Use inline comments for complex logic with `#` prefix
- **Print Logging**: Use structured print statements with `[TAG]` prefixes:
  - `[INFO]` for general information
  - `[ERROR]` for errors
  - `[SUCCESS]` for successful operations
  - `[WARNING]` for warnings
  - `[INIT]` for initialization

Example:
```python
def calculate_geo_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate Haversine distance between two coordinates.
    
    Args:
        lon1: Longitude of first point
        lat1: Latitude of first point
        lon2: Longitude of second point
        lat2: Latitude of second point
    
    Returns:
        Distance in kilometers
    """
    # Implementation here
```

### JavaScript/React Code Style

- Use functional components with hooks
- Destructure props in component parameters
- Use Material-UI `sx` prop for styling
- Prefix custom markers and CSS classes appropriately

---

## Testing Instructions

### Manual Testing Workflow

1. **Set API Keys**: Set `BODS_API_KEY` environment variable (required)
2. **Start Backend**: `python server.py` - Verify it loads the ML model and starts on port 8000
3. **Check Health**: Visit `http://localhost:8000/api/health` to verify API configuration
4. **Start Frontend**: `npm run dev` in frontend directory - Verify it compiles without errors
5. **Test API**: Visit `http://localhost:8000/docs` to test endpoints via Swagger UI
6. **Test Map Interface**: Open `http://localhost:5173` and verify:
   - Map loads centered on Bristol
   - Station markers are visible
   - Clicking a station shows approaching buses with real-time GPS data
   - Vehicle tracking displays correctly

### Component Testing

- **CV Module**: Run `python pedestrian_detector.py --station STATION_TM01 --video crowd_busy.mp4` and verify YOLO detects people
- **ML Model**: Run `python model_trainer.py` and verify model trains with MAE < 3.0 minutes

---

## Security Considerations

### API Keys

API keys should be set via environment variables:

```bash
# Required
export BODS_API_KEY="your_bods_api_key_here"

# Optional (for traffic data)
export TOMTOM_API_KEY="your_tomtom_api_key_here"
```

**DO NOT** hardcode API keys in the source code.

To get a BODS API key:
1. Register at https://data.bus-data.dft.gov.uk/
2. Go to your account settings
3. Generate an API key

To get a TomTom API key:
1. Register at https://developer.tomtom.com/
2. Create a new app
3. Copy the API key (free tier: 2,500 requests/day)

### CORS Configuration

The backend allows all origins (`["*"]`) for development convenience. Restrict this in production:
```python
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Specific origin
    ...
)
```

### Input Validation

- FastAPI automatically validates request bodies using Pydantic models
- Station IDs are validated against the `TRANSIT_STATIONS` dictionary
- Coordinate bounds are checked against Bristol area

---

## Development Notes

### Data Flow

1. **Bus Location**: Fetched from BODS API every 10-20 seconds (cached)
2. **Traffic Data**: Real-time from TomTom API only (no fallback simulation)
3. **Crowd Count**: From CV module via `/api/sensors/update` or defaults to 0
4. **ML Prediction**: XGBoost model combines factors to predict delay
5. **ETA Calculation**: Base travel time + predicted delay

### No Simulated Data

The system does NOT use simulated bus data. If the BODS API:
- Returns no data → Empty result is returned
- Fails → Empty result is returned
- Times out → Empty result is returned

You must configure a valid `BODS_API_KEY` to see real-time bus data.

### Station Configuration

Stations are defined in the `TRANSIT_STATIONS` dictionary in `server.py`. Each station has:
- `display_name`: Human-readable name
- `position`: lat/lon coordinates
- `atco_reference`: ATCO code for API lookup
- `serving_route`: Route number (72, N1, N86)
- `sequence_number`: Stop order on route

### ML Model Features

The model uses three features for prediction:
- `traffic_delay`: Minutes of traffic delay
- `crowd_count`: Number of waiting passengers
- `is_raining`: Weather condition (0 or 1)

Target variable: `actual_arrival_time` (minutes delayed)

---

## Common Issues

1. **No buses showing**: Check that `BODS_API_KEY` environment variable is set correctly
2. **YOLO model not found**: Ensure `yolo.pt` is in the `Bus Predictor` directory
3. **CV module can't find video**: Place videos in `Bus Predictor/videos/` directory
4. **Frontend can't connect to backend**: Verify backend is running on port 8000
5. **CORS errors**: Check CORS middleware configuration in `server.py`
6. **"NOT CONFIGURED" in health check**: Set the required environment variables

---

## Credits

- Bus Data: Bus Open Data Service (BODS)
- Traffic Data: TomTom Developer
- Computer Vision: Ultralytics YOLOv8
- Maps: OpenStreetMap + Leaflet
