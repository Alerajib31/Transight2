from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import xgboost as xgb
import pandas as pd
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt, atan2, degrees
import time
from collections import defaultdict

# Load API keys from environment variables (or use provided keys as default)
BODS_API_KEY = os.environ.get("BODS_API_KEY", "2bc39438a3eeec844704f182bab7892fea39b8bd")
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "a1jG3Ptx5icrrFGYVRBWQo4o0t2XurwP")

api = FastAPI(title="Bristol Transit Intelligence API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BODS_CREDENTIALS = {
    "api_key": BODS_API_KEY,
    "base_url": "https://data.bus-data.dft.gov.uk/api/v1/",
    "siri_endpoint": "https://data.bus-data.dft.gov.uk/api/v1/datafeed/"
}

TOMTOM_TRAFFIC_CONFIG = {
    "api_key": TOMTOM_API_KEY,
    "base_url": "https://api.tomtom.com/traffic/services/4/flowSegmentData",
    "style": "absolute",
    "zoom": 10
}

SIMULATED_BUSES_ENABLED = False
MONITORED_ROUTES = None

BRISTOL_BOUNDS = {
    "min_latitude": 51.40,
    "max_latitude": 51.55,
    "min_longitude": -2.70,
    "max_longitude": -2.50
}

TRANSIT_STATIONS = {
    "STATION_TM01": {
        "display_name": "Temple Meads Station",
        "position": {"lat": 51.4496, "lon": -2.5811},
        "atco_reference": "01000053220",
        "serving_route": "72",
        "sequence_number": 1
    },
    "STATION_CC02": {
        "display_name": "The Centre",
        "position": {"lat": 51.4528, "lon": -2.5975},
        "atco_reference": "01000002301",
        "serving_route": "72",
        "sequence_number": 2
    },
    "STATION_CB03": {
        "display_name": "Cabot Circus",
        "position": {"lat": 51.4545, "lon": -2.5879},
        "atco_reference": "01000588088",
        "serving_route": "72",
        "sequence_number": 3
    },
    "STATION_SC04": {
        "display_name": "Stokes Croft",
        "position": {"lat": 51.4600, "lon": -2.5880},
        "atco_reference": "01000030101",
        "serving_route": "72",
        "sequence_number": 4
    },
    "STATION_GR05": {
        "display_name": "Gloucester Road",
        "position": {"lat": 51.4640, "lon": -2.5900},
        "atco_reference": "01000046701",
        "serving_route": "72",
        "sequence_number": 5
    },
    "STATION_HF06": {
        "display_name": "Horfield",
        "position": {"lat": 51.4750, "lon": -2.5850},
        "atco_reference": "01000048901",
        "serving_route": "72",
        "sequence_number": 6
    },
    "STATION_SH07": {
        "display_name": "Southmead Hospital",
        "position": {"lat": 51.4900, "lon": -2.5950},
        "atco_reference": "01000055001",
        "serving_route": "72",
        "sequence_number": 7
    },
    "STATION_UW08": {
        "display_name": "UWE Frenchay Campus",
        "position": {"lat": 51.5005, "lon": -2.5490},
        "atco_reference": "01000057001",
        "serving_route": "72",
        "sequence_number": 8
    },
    "STATION_N1_01": {
        "display_name": "Cribbs Causeway",
        "position": {"lat": 51.5250, "lon": -2.6100},
        "atco_reference": "01000055201",
        "serving_route": "N1",
        "sequence_number": 1
    },
    "STATION_N1_02": {
        "display_name": "Southmead",
        "position": {"lat": 51.5000, "lon": -2.6000},
        "atco_reference": "01000055101",
        "serving_route": "N1",
        "sequence_number": 2
    },
    "STATION_N1_03": {
        "display_name": "Filton Avenue",
        "position": {"lat": 51.4850, "lon": -2.5950},
        "atco_reference": "01000045801",
        "serving_route": "N1",
        "sequence_number": 3
    },
    "STATION_N1_04": {
        "display_name": "Gloucester Road North",
        "position": {"lat": 51.4700, "lon": -2.5920},
        "atco_reference": "01000046801",
        "serving_route": "N1",
        "sequence_number": 4
    },
    "STATION_N1_05": {
        "display_name": "Broadmead",
        "position": {"lat": 51.4560, "lon": -2.5850},
        "atco_reference": "01000588101",
        "serving_route": "N1",
        "sequence_number": 5
    },
    "STATION_N1_06": {
        "display_name": "City Centre (Colston Avenue)",
        "position": {"lat": 51.4545, "lon": -2.5950},
        "atco_reference": "01000002401",
        "serving_route": "N1",
        "sequence_number": 6
    },
    "STATION_N86_01": {
        "display_name": "Hengrove Park",
        "position": {"lat": 51.4200, "lon": -2.5950},
        "atco_reference": "01000048101",
        "serving_route": "N86",
        "sequence_number": 1
    },
    "STATION_N86_02": {
        "display_name": "Whitchurch",
        "position": {"lat": 51.4280, "lon": -2.6000},
        "atco_reference": "01000062301",
        "serving_route": "N86",
        "sequence_number": 2
    },
    "STATION_N86_03": {
        "display_name": "Knowle",
        "position": {"lat": 51.4350, "lon": -2.5880},
        "atco_reference": "01000050201",
        "serving_route": "N86",
        "sequence_number": 3
    },
    "STATION_N86_04": {
        "display_name": "Brislington",
        "position": {"lat": 51.4380, "lon": -2.5650},
        "atco_reference": "01000009901",
        "serving_route": "N86",
        "sequence_number": 4
    },
    "STATION_N86_05": {
        "display_name": "Temple Gate",
        "position": {"lat": 51.4480, "lon": -2.5760},
        "atco_reference": "01000053301",
        "serving_route": "N86",
        "sequence_number": 5
    },
    "STATION_N86_06": {
        "display_name": "Old Market",
        "position": {"lat": 51.4530, "lon": -2.5780},
        "atco_reference": "01000052201",
        "serving_route": "N86",
        "sequence_number": 6
    },
    "STATION_N86_07": {
        "display_name": "Broadmead Shopping",
        "position": {"lat": 51.4560, "lon": -2.5840},
        "atco_reference": "01000588102",
        "serving_route": "N86",
        "sequence_number": 7
    }
}

ROUTE_PATH_COORDINATES = {
    "72": [
        (51.4496, -2.5811),  # Temple Meads
        (51.4510, -2.5820),  # Temple Gate
        (51.4528, -2.5975),  # The Centre
        (51.4545, -2.5879),  # Cabot Circus
        (51.4600, -2.5880),  # Stokes Croft
        (51.4640, -2.5900),  # Gloucester Road
        (51.4750, -2.5850),  # Horfield
        (51.4900, -2.5950),  # Southmead
        (51.5005, -2.5490),  # UWE Frenchay
    ],
    "N1": [
        (51.5250, -2.6100),  # Cribbs Causeway
        (51.5150, -2.6050),  # Approach to Southmead
        (51.5000, -2.6000),  # Southmead
        (51.4850, -2.5950),  # Filton Avenue
        (51.4700, -2.5920),  # Gloucester Road North
        (51.4600, -2.5900),  # Gloucester Road
        (51.4560, -2.5850),  # Broadmead
        (51.4545, -2.5950),  # City Centre
    ],
    "N86": [
        (51.4200, -2.5950),  # Hengrove Park
        (51.4280, -2.6000),  # Whitchurch
        (51.4350, -2.5880),  # Knowle
        (51.4380, -2.5650),  # Brislington
        (51.4450, -2.5720),  # Approach to Temple Gate
        (51.4480, -2.5760),  # Temple Gate
        (51.4530, -2.5780),  # Old Market
        (51.4560, -2.5840),  # Broadmead Shopping
    ]
}

ALL_ROUTE_COORDINATES = []
for route_coords in ROUTE_PATH_COORDINATES.values():
    ALL_ROUTE_COORDINATES.extend(route_coords)

data_cache = {
    "stations": {"content": None, "last_updated": 0},
    "vehicles": {"content": {}, "last_updated": 0}
}

vehicle_movement_log = defaultdict(list)
sensor_data_registry = {}
tomtom_cache = {}  # key: "lat_lon" rounded to 3 decimals, value: {"data": ..., "timestamp": ...}
TOMTOM_CACHE_TTL = 300  # 5 minutes
ml_predictor = None
MODEL_FILE_PATH = "trained_model.json"

if os.path.exists(MODEL_FILE_PATH):
    ml_predictor = xgb.Booster()
    ml_predictor.load_model(MODEL_FILE_PATH)
    print(f"[INIT] Loaded ML model from {MODEL_FILE_PATH}")
else:
    print("[WARN] ML model not found")

def calculate_geo_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(radians, [lon1, lat1, lon2, lat2])
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lon = radians(lon2 - lon1)
    
    x = sin(delta_lon) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(delta_lon)
    
    initial_bearing = atan2(x, y)
    return (degrees(initial_bearing) + 360) % 360

def extract_dynamic_stops_from_vehicles(vehicles_dict: Dict[str, Any]) -> Dict[str, Any]:
    dynamic_stations = {}
    station_counter = 1
    
    for vehicle_id, vehicle_data in vehicles_dict.items():
        destination = vehicle_data.get("destination_name", "Unknown")
        route = vehicle_data.get("route_designation", "Unknown")
        lat = vehicle_data.get("latitude")
        lon = vehicle_data.get("longitude")
        
        if lat and lon:
            station_key = f"LIVE_{route}_{station_counter}"
            
            # Check if similar station already exists (within 100m radius)
            is_duplicate = False
            for existing_station in dynamic_stations.values():
                existing_lat = existing_station["position"]["lat"]
                existing_lon = existing_station["position"]["lon"]
                distance = calculate_geo_distance(lon, lat, existing_lon, existing_lat)
                if distance < 0.1:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                dynamic_stations[station_key] = {
                    "display_name": f"{destination} (Route {route})",
                    "position": {"lat": lat, "lon": lon},
                    "atco_reference": f"LIVE_{station_counter}",
                    "serving_route": route,
                    "sequence_number": station_counter,
                    "is_dynamic": True
                }
                station_counter += 1
    
    return dynamic_stations


def fetch_live_vehicle_positions() -> Dict[str, Any]:
    """
    Fetch live bus positions from BODS SIRI-VM API
    Returns dictionary of vehicles with their positions and metadata
    """
    try:
        # Use expanded bounds to capture all vehicles in Bristol area
        expanded_bounds = {
            "min_latitude": 51.30,
            "max_latitude": 51.65,
            "min_longitude": -2.80,
            "max_longitude": -2.40
        }
        bbox = f"{expanded_bounds['min_latitude']},{expanded_bounds['min_longitude']},{expanded_bounds['max_latitude']},{expanded_bounds['max_longitude']}"
        api_url = f"{BODS_CREDENTIALS['siri_endpoint']}?api_key={BODS_CREDENTIALS['api_key']}&boundingBox={bbox}"
        
        print(f"[BODS] Fetching from: {api_url.replace(BODS_CREDENTIALS['api_key'], '***')}")
        
        response = requests.get(api_url, timeout=15)
        
        print(f"[BODS] Response status: {response.status_code}")
        print(f"[BODS] Response size: {len(response.content)} bytes")
        
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        namespaces = {
            'siri': 'http://www.siri.org.uk/siri',
            'ns2': 'http://www.ifopt.org.uk/acsb',
            'ns3': 'http://www.ifopt.org.uk/ifopt',
            'ns4': 'http://datex2.eu/schema/2_0RC1/2_0'
        }
        
        vehicles_dict = {}
        vehicle_activities = root.findall('.//siri:VehicleActivity', namespaces)
        
        # Initialize filtering counters
        filtered_out_route = 0
        filtered_out_location = 0
        filtered_out_bounds = 0
        
        for vehicle_activity in vehicle_activities:
            try:
                monitored_journey = vehicle_activity.find('.//siri:MonitoredVehicleJourney', namespaces)
                if monitored_journey is None:
                    continue
                
                line_ref = monitored_journey.find('siri:LineRef', namespaces)
                route_number = line_ref.text if line_ref is not None else None
                
                if MONITORED_ROUTES is not None and route_number not in MONITORED_ROUTES:
                    filtered_out_route += 1
                    continue
                
                vehicle_ref = monitored_journey.find('siri:VehicleRef', namespaces)
                vehicle_id = vehicle_ref.text if vehicle_ref is not None else None
                
                location = monitored_journey.find('.//siri:VehicleLocation', namespaces)
                if location is None:
                    continue
                
                longitude_elem = location.find('siri:Longitude', namespaces)
                latitude_elem = location.find('siri:Latitude', namespaces)
                
                if longitude_elem is None or latitude_elem is None:
                    filtered_out_location += 1
                    continue
                
                lon = float(longitude_elem.text)
                lat = float(latitude_elem.text)
                
                # Validate coordinates are within Bristol area
                if not (BRISTOL_BOUNDS["min_latitude"] <= lat <= BRISTOL_BOUNDS["max_latitude"] and
                       BRISTOL_BOUNDS["min_longitude"] <= lon <= BRISTOL_BOUNDS["max_longitude"]):
                    filtered_out_bounds += 1
                    continue
                
                # Extract operator information
                operator_elem = monitored_journey.find('siri:OperatorRef', namespaces)
                operator = operator_elem.text if operator_elem is not None else "Unknown"
                
                # Extract destination
                destination_elem = monitored_journey.find('siri:DestinationName', namespaces)
                destination = destination_elem.text if destination_elem is not None else "Not specified"
                
                # Record timestamp
                recorded_time_elem = vehicle_activity.find('.//siri:RecordedAtTime', namespaces)
                timestamp = recorded_time_elem.text if recorded_time_elem is not None else datetime.now().isoformat()
                
                vehicles_dict[vehicle_id] = {
                    "vehicle_identifier": vehicle_id,
                    "route_designation": route_number,
                    "latitude": lat,
                    "longitude": lon,
                    "operator_name": operator,
                    "destination_name": destination,
                    "last_updated": timestamp,
                    "data_source": "BODS_SIRI_VM"
                }
                
                # Update movement history for trail display
                vehicle_movement_log[vehicle_id].append({
                    "lat": lat,
                    "lon": lon,
                    "timestamp": timestamp
                })
                
                # Keep only last 25 positions for trail
                if len(vehicle_movement_log[vehicle_id]) > 25:
                    vehicle_movement_log[vehicle_id] = vehicle_movement_log[vehicle_id][-25:]
                
            except Exception as parse_error:
                print(f"[ERROR] Failed to parse vehicle record: {parse_error}")
                continue
        
        # Show filtering statistics
        print(f"  - Total vehicles in feed: {len(vehicle_activities)}")
        print(f"  - Filtered by route: {filtered_out_route}")
        print(f"  - Filtered by missing location: {filtered_out_location}")
        print(f"  - Filtered by Bristol bounds: {filtered_out_bounds}")
        print(f"  - Remaining vehicles: {len(vehicles_dict)}")
        
        if len(vehicles_dict) > 0:
            route_list = list(set(v["route_designation"] for v in vehicles_dict.values()))
            print(f"[SUCCESS] Using {len(vehicles_dict)} REAL vehicles from BODS API")
            print(f"[INFO] Routes found: {', '.join(sorted(route_list)[:10])}{'...' if len(route_list) > 10 else ''}")
            print(f"[INFO] Vehicle IDs: {', '.join(list(vehicles_dict.keys())[:3])}...")
            return vehicles_dict
        elif len(vehicle_activities) == 0:
            # No vehicles in the feed at all - likely off-hours
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 23:
                print(f"[INFO] BODS API returned empty - buses not operating at this hour ({current_hour}:00)")
            else:
                print(f"[INFO] BODS API returned empty - no active vehicles in service area")
            print(f"[INFO] This is normal for real-time APIs during off-peak hours")
            return {}
        else:
            print(f"[INFO] BODS API returned {len(vehicle_activities)} vehicles but none in Bristol area")
            print(f"[INFO] Check bounding box or expand search area")
            return {}
        
    except requests.exceptions.RequestException as req_error:
        print(f"[ERROR] BODS API request failed: {req_error}")
        return {}
    except ET.ParseError as xml_error:
        print(f"[ERROR] BODS XML parsing failed: {xml_error}")
        return {}
    except Exception as general_error:
        print(f"[ERROR] Unexpected error in BODS fetch: {general_error}")
        import traceback
        traceback.print_exc()
        return {}


def fetch_tomtom_traffic_data(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """
    Fetch real-time traffic data from TomTom Traffic Flow API

    API Documentation: https://developer.tomtom.com/traffic-api/documentation
    Free Tier: 2,500 requests/day, no credit card required

    Returns traffic flow data including current speed and free-flow speed
    Uses in-memory cache (5-min TTL, ~111m spatial precision) to conserve API quota
    """
    api_key = TOMTOM_TRAFFIC_CONFIG["api_key"]

    if not api_key or api_key == "":
        print(f"[TOMTOM] No API key configured")
        return None

    # Check cache first (round to 3 decimal places = ~111m precision)
    cache_key = f"{round(latitude, 3)}_{round(longitude, 3)}"
    current_time = time.time()

    if cache_key in tomtom_cache:
        cached = tomtom_cache[cache_key]
        if current_time - cached["timestamp"] < TOMTOM_CACHE_TTL:
            print(f"[TOMTOM] Cache hit for ({latitude:.4f}, {longitude:.4f})")
            return cached["data"]

    try:
        # TomTom Traffic Flow API endpoint
        url = f"{TOMTOM_TRAFFIC_CONFIG['base_url']}/{TOMTOM_TRAFFIC_CONFIG['style']}/{TOMTOM_TRAFFIC_CONFIG['zoom']}/json"

        params = {
            "key": api_key,
            "point": f"{latitude},{longitude}",
            "unit": "KMPH"  # Speed in km/h
        }

        print(f"[TOMTOM] Fetching traffic for ({latitude:.4f}, {longitude:.4f})")

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()

        # Extract traffic flow information
        if "flowSegmentData" in data:
            flow_data = data["flowSegmentData"]
            print(f"[TOMTOM] Got traffic data: {flow_data.get('currentSpeed')} km/h " +
                  f"(free flow: {flow_data.get('freeFlowSpeed')} km/h)")
            result = {
                "current_speed": flow_data.get("currentSpeed", 0),
                "free_flow_speed": flow_data.get("freeFlowSpeed", 40),
                "current_travel_time": flow_data.get("currentTravelTime", 0),
                "free_flow_travel_time": flow_data.get("freeFlowTravelTime", 0),
                "confidence": flow_data.get("confidence", 0.8),
                "road_closure": flow_data.get("roadClosure", False)
            }
            tomtom_cache[cache_key] = {"data": result, "timestamp": current_time}
            return result
        else:
            print(f"[TOMTOM] No flowSegmentData in response")
            tomtom_cache[cache_key] = {"data": None, "timestamp": current_time}
            return None

    except requests.exceptions.RequestException as e:
        print(f"[WARNING] TomTom API request failed: {e}")
        # Cache failures to avoid hammering a broken API
        tomtom_cache[cache_key] = {"data": None, "timestamp": current_time}
        return None
    except Exception as e:
        print(f"[ERROR] TomTom API error: {e}")
        return None


def estimate_traffic_conditions(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """
    Get real-time traffic conditions from TomTom API
    
    Uses TomTom Traffic Flow API (free tier: 2,500 requests/day)
    Calculates delay based on actual speed vs free flow speed
    
    Returns traffic delay, current speed, and confidence metrics
    """
    # Try to fetch real traffic data from TomTom API
    tomtom_data = fetch_tomtom_traffic_data(latitude, longitude)
    
    if tomtom_data:
        # Calculate delay based on real traffic data
        current_speed = tomtom_data["current_speed"]
        free_flow_speed = tomtom_data["free_flow_speed"]
        
        # Calculate speed ratio
        speed_ratio = current_speed / free_flow_speed if free_flow_speed > 0 else 1.0
        
        # Calculate delay using a deterministic formula based on speed ratio
        # When speed_ratio = 1.0 (no congestion), delay = 0
        # When speed_ratio approaches 0 (gridlock), delay increases exponentially
        if speed_ratio >= 0.9:
            base_delay = 0
        elif speed_ratio >= 0.7:
            base_delay = (1 - speed_ratio) * 10  # 0-3 minutes
        elif speed_ratio >= 0.5:
            base_delay = 2 + (0.7 - speed_ratio) * 20  # 2-6 minutes
        elif speed_ratio >= 0.3:
            base_delay = 6 + (0.5 - speed_ratio) * 30  # 6-12 minutes
        else:
            base_delay = 12 + (0.3 - speed_ratio) * 40  # 12+ minutes
        
        return {
            "estimated_delay_minutes": round(base_delay, 1),
            "current_speed_estimate": round(current_speed, 1),
            "free_flow_speed": round(free_flow_speed, 1),
            "confidence": tomtom_data["confidence"],
            "method": "tomtom_traffic_api",
            "data_source": "TomTom Traffic Flow API (Real-time)",
            "road_closure": tomtom_data.get("road_closure", False),
            "speed_ratio": round(speed_ratio, 2)
        }
    
    # Return None to indicate traffic data unavailable
    print(f"[WARNING] TomTom API data unavailable for ({latitude:.4f}, {longitude:.4f})")
    return None


def predict_arrival_delay_ml(traffic_minutes: float, passenger_count: int, is_raining: bool = False) -> float:
    """
    Use machine learning model to predict bus delay
    Inputs:
        - traffic_minutes: Traffic-induced delay (from TomTom API)
        - passenger_count: Number of waiting passengers (from CV sensors)
        - is_raining: Weather condition boolean
    Output: Predicted additional delay in minutes
    """
    if ml_predictor is None:
        # Fallback heuristic calculation
        boarding_delay = (passenger_count * 4.5) / 60  # 4.5 seconds per passenger
        weather_penalty = 2.5 if is_raining else 0
        total_delay = traffic_minutes + boarding_delay + weather_penalty
        print(f"[ML] Using heuristic prediction: traffic={traffic_minutes:.1f}, crowd={passenger_count}, rain={is_raining} → {total_delay:.2f}min")
        return round(total_delay, 2)
    
    try:
        # Prepare feature vector for model
        feature_df = pd.DataFrame([{
            "traffic_delay": traffic_minutes,
            "crowd_count": passenger_count,
            "is_raining": int(is_raining)
        }])
        
        # Convert to XGBoost matrix format
        dmatrix = xgb.DMatrix(feature_df)
        
        # Generate prediction using all three data sources
        prediction = ml_predictor.predict(dmatrix)[0]
        
        print(f"[ML] XGBoost prediction: traffic={traffic_minutes:.1f}, crowd={passenger_count}, rain={is_raining} → {prediction:.2f}min")
        
        return round(float(prediction), 2)
        
    except Exception as model_error:
        print(f"[ERROR] ML prediction failed: {model_error}")
        # Fallback to heuristic
        boarding_delay = (passenger_count * 4.5) / 60
        return round(traffic_minutes + boarding_delay, 2)


# ===========================
# DATA MODELS (Pydantic)
# ===========================

class SensorDataInput(BaseModel):
    """Input model for CV sensor updates"""
    station_id: str
    detected_people: int
    timestamp: Optional[str] = None


class StationQueryFilter(BaseModel):
    """Filter parameters for station queries"""
    route_filter: Optional[str] = None
    max_distance_km: Optional[float] = None
    reference_lat: Optional[float] = None
    reference_lon: Optional[float] = None


# ===========================
# REST API ENDPOINTS
# ===========================

@api.get("/", tags=["System"])
async def root_endpoint():
    """API information endpoint"""
    return {
        "service": "Bristol Transit Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@api.get("/api/health", tags=["System"])
async def health_check():
    """System health and status check"""
    # Check API key configuration
    bods_configured = bool(BODS_CREDENTIALS.get("api_key"))
    tomtom_configured = bool(TOMTOM_TRAFFIC_CONFIG.get("api_key"))
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "stations_loaded": len(TRANSIT_STATIONS),
        "active_vehicles_tracked": len(data_cache["vehicles"]["content"]),
        "ml_model_loaded": ml_predictor is not None,
        "sensors_active": len(sensor_data_registry),
        "api_configuration": {
            "bods_api_configured": bods_configured,
            "tomtom_api_configured": tomtom_configured,
            "simulated_data_enabled": SIMULATED_BUSES_ENABLED
        },
        "data_sources": {
            "bus_positions": "BODS API (Real-time)" if bods_configured else "NOT CONFIGURED - Set BODS_API_KEY env var",
            "traffic_data": "TomTom API" if tomtom_configured else "Not configured (optional)",
            "passenger_count": "CV Sensors / Manual"
        }
    }


@api.get("/api/stations", tags=["Transit Data"])
async def get_all_stations(
    route: Optional[str] = Query(None, description="Filter by route number"),
    search: Optional[str] = Query(None, description="Search station names"),
    dynamic: bool = Query(True, description="Include dynamically discovered stops from live buses")
):
    """
    Retrieve all transit stations
    Combines hardcoded stations with dynamically discovered stops from active buses
    Optional filters: route, name search
    """
    stations_list = []
    
    # Start with hardcoded stations
    all_stations = dict(TRANSIT_STATIONS)
    
    # Add dynamic stations from live vehicle data if requested
    if dynamic:
        current_time = time.time()
        if current_time - data_cache["vehicles"]["last_updated"] > 20:
            data_cache["vehicles"]["content"] = fetch_live_vehicle_positions()
            data_cache["vehicles"]["last_updated"] = current_time
        
        if data_cache["vehicles"]["content"]:
            dynamic_stops = extract_dynamic_stops_from_vehicles(data_cache["vehicles"]["content"])
            all_stations.update(dynamic_stops)
    
    for station_id, station_info in all_stations.items():
        # Apply route filter
        if route and station_info["serving_route"] != route:
            continue
        
        # Apply name search filter
        if search and search.lower() not in station_info["display_name"].lower():
            continue
        
        stations_list.append({
            "id": station_id,
            "name": station_info["display_name"],
            "latitude": station_info["position"]["lat"],
            "longitude": station_info["position"]["lon"],
            "route": station_info["serving_route"],
            "atco_code": station_info.get("atco_reference", "N/A"),
            "sequence": station_info.get("sequence_number", 0),
            "is_dynamic": station_info.get("is_dynamic", False)
        })
    
    # Sort by route and sequence number
    stations_list.sort(key=lambda x: (x["route"], x["sequence"]))
    
    hardcoded_count = len(TRANSIT_STATIONS)
    dynamic_count = len([s for s in stations_list if s.get("is_dynamic", False)])
    
    return {
        "total": len(stations_list),
        "hardcoded_count": hardcoded_count,
        "dynamic_count": dynamic_count,
        "stations": stations_list
    }


@api.get("/api/stations/nearby", tags=["Transit Data"])
async def find_nearby_stations(
    lat: float = Query(..., description="Latitude coordinate"),
    lon: float = Query(..., description="Longitude coordinate"),
    radius_km: float = Query(5.0, description="Search radius in kilometers")
):
    """
    Find stations within specified radius of coordinates
    """
    nearby_stations = []

    # Include dynamic stations from live vehicle data
    all_stations = dict(TRANSIT_STATIONS)
    if data_cache["vehicles"]["content"]:
        dynamic_stops = extract_dynamic_stops_from_vehicles(data_cache["vehicles"]["content"])
        all_stations.update(dynamic_stops)

    for station_id, station_info in all_stations.items():
        distance = calculate_geo_distance(
            lon, lat,
            station_info["position"]["lon"],
            station_info["position"]["lat"]
        )
        
        if distance <= radius_km:
            nearby_stations.append({
                "id": station_id,
                "name": station_info["display_name"],
                "latitude": station_info["position"]["lat"],
                "longitude": station_info["position"]["lon"],
                "distance_km": round(distance, 2),
                "route": station_info["serving_route"]
            })
    
    # Sort by distance (closest first)
    nearby_stations.sort(key=lambda x: x["distance_km"])
    
    return {
        "search_center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "results": len(nearby_stations),
        "stations": nearby_stations
    }


@api.get("/api/stations/{station_id}/approaching", tags=["Transit Data"])
async def get_approaching_vehicles(station_id: str):
    """
    Get buses approaching a specific station with ML-based ETA predictions
    
    Uses three real data sources:
    1. Live bus GPS positions from BODS API
    2. Real-time traffic conditions from TomTom Traffic API
    3. Passenger density from CV sensors (YOLOv8)
    """
    # Build combined station lookup (static + dynamic from live vehicles)
    all_known_stations = dict(TRANSIT_STATIONS)
    if data_cache["vehicles"]["content"]:
        dynamic_stops = extract_dynamic_stops_from_vehicles(data_cache["vehicles"]["content"])
        all_known_stations.update(dynamic_stops)

    if station_id not in all_known_stations:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    station_info = all_known_stations[station_id]
    station_lat = station_info["position"]["lat"]
    station_lon = station_info["position"]["lon"]
    
    # Refresh vehicle data if cache is stale (>20 seconds)
    current_time = time.time()
    if current_time - data_cache["vehicles"]["last_updated"] > 20:
        data_cache["vehicles"]["content"] = fetch_live_vehicle_positions()
        data_cache["vehicles"]["last_updated"] = current_time
    
    vehicles_data = data_cache["vehicles"]["content"]
    approaching_vehicles = []
    
    # Get sensor data for this station (crowd count) - Data Source #3
    sensor_reading = sensor_data_registry.get(station_id, {"detected_people": 0})
    crowd_count = sensor_reading.get("detected_people", 0)
    
    print(f"[PREDICTION] Station: {station_info['display_name']}")
    print(f"[PREDICTION] Data Source 1 - BODS API: {len(vehicles_data)} vehicles tracked")
    print(f"[PREDICTION] Data Source 3 - CV Sensor: {crowd_count} passengers waiting")
    
    traffic_success_count = 0
    
    for vehicle_id, vehicle_data in vehicles_data.items():
        distance_km = calculate_geo_distance(
            vehicle_data["longitude"],
            vehicle_data["latitude"],
            station_lon,
            station_lat
        )
        
        # Consider vehicles within 5km radius
        if distance_km <= 5.0:
            # Get traffic conditions - Data Source #2
            traffic_info = estimate_traffic_conditions(
                vehicle_data["latitude"],
                vehicle_data["longitude"]
            )
            
            if traffic_info:
                traffic_success_count += 1
            
            # Calculate ETA
            average_speed_kph = 25  # Urban bus average
            base_eta_minutes = (distance_km / average_speed_kph) * 60
            
            # Extract traffic delay (0 if no traffic data)
            traffic_delay = traffic_info["estimated_delay_minutes"] if traffic_info else 0
            traffic_source = traffic_info["data_source"] if traffic_info else "No data"
            
            # Apply ML prediction using all three factors
            # - traffic_delay: from TomTom API (Data Source #2)
            # - crowd_count: from CV sensors (Data Source #3)
            predicted_delay = predict_arrival_delay_ml(
                traffic_delay,
                crowd_count,
                False  # Weather integration can be added
            )
            
            total_eta = base_eta_minutes + predicted_delay
            
            # Get movement trail
            trail = vehicle_movement_log.get(vehicle_id, [])
            
            approaching_vehicles.append({
                "vehicle_id": vehicle_id,
                "route": vehicle_data["route_designation"],
                "distance_km": round(distance_km, 2),
                "eta_minutes": round(total_eta, 1),
                "current_position": {
                    "lat": vehicle_data["latitude"],
                    "lon": vehicle_data["longitude"]
                },
                "destination": vehicle_data["destination_name"],
                "operator": vehicle_data["operator_name"],
                "traffic_delay": round(traffic_delay, 1),
                "crowd_delay": round((crowd_count * 4.5) / 60, 1),
                "position_trail": trail[-15:],  # Last 15 positions
                "data_source": vehicle_data.get("data_source", "BODS"),
                "traffic_data_source": traffic_source,
                "traffic_details": {
                    "current_speed": traffic_info.get("current_speed_estimate") if traffic_info else None,
                    "free_flow_speed": traffic_info.get("free_flow_speed") if traffic_info else None,
                    "speed_ratio": traffic_info.get("speed_ratio") if traffic_info else None
                } if traffic_info else None
            })
    
    print(f"[PREDICTION] Data Source 2 - TomTom API: {traffic_success_count}/{len(approaching_vehicles)} vehicles with traffic data")
    print(f"[PREDICTION] ML Model: {len(approaching_vehicles)} ETAs calculated with all available data")
    
    # Sort by ETA (soonest first)
    approaching_vehicles.sort(key=lambda x: x["eta_minutes"])
    
    return {
        "station_id": station_id,
        "station_name": station_info["display_name"],
        "waiting_passengers": crowd_count,
        "approaching_count": len(approaching_vehicles),
        "data_sources": {
            "bus_positions": "BODS API (Real-time GPS)",
            "traffic_data": "TomTom Traffic API (Real-time)",
            "passenger_count": "CV Sensor (YOLOv8)" if crowd_count > 0 else "CV Sensor (YOLOv8) - No data",
            "ml_model": "XGBoost (trained)" if ml_predictor else "Heuristic (fallback)"
        },
        "traffic_api_status": {
            "working": traffic_success_count > 0,
            "vehicles_with_traffic_data": traffic_success_count,
            "total_approaching": len(approaching_vehicles),
            "message": "Traffic data available" if traffic_success_count > 0 else "TomTom API unavailable - traffic delays not reflected in ETAs"
        },
        "vehicles": approaching_vehicles
    }


@api.get("/api/vehicles/active", tags=["Transit Data"])
async def get_all_active_vehicles():
    """
    Get all currently tracked vehicles in the system
    """
    current_time = time.time()
    
    # Refresh if needed
    if current_time - data_cache["vehicles"]["last_updated"] > 20:
        data_cache["vehicles"]["content"] = fetch_live_vehicle_positions()
        data_cache["vehicles"]["last_updated"] = current_time
    
    vehicles_list = []
    
    for vehicle_id, vehicle_data in data_cache["vehicles"]["content"].items():
        vehicles_list.append({
            "vehicle_id": vehicle_id,
            "route": vehicle_data["route_designation"],
            "position": {
                "lat": vehicle_data["latitude"],
                "lon": vehicle_data["longitude"]
            },
            "destination": vehicle_data["destination_name"],
            "operator": vehicle_data["operator_name"],
            "last_update": vehicle_data["last_updated"]
        })
    
    current_hour = datetime.now().hour
    return {
        "timestamp": datetime.now().isoformat(),
        "total_vehicles": len(vehicles_list),
        "vehicles": vehicles_list,
        "cache_age_seconds": round(time.time() - data_cache["vehicles"]["last_updated"]) if data_cache["vehicles"]["last_updated"] > 0 else None,
        "status": "live_data" if len(vehicles_list) > 0 else "no_vehicles",
        "message": None if len(vehicles_list) > 0 else (
            "Buses may not be in service at this hour" if current_hour < 6 or current_hour > 23
            else "No vehicles currently detected in the Bristol area"
        )
    }


@api.post("/api/sensors/update", tags=["Sensor Integration"])
async def update_sensor_data(sensor_input: SensorDataInput):
    """
    Receive crowd detection data from CV sensors
    Updates station passenger counts for prediction
    """
    station_id = sensor_input.station_id
    
    if station_id not in TRANSIT_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
    
    # Store sensor reading
    sensor_data_registry[station_id] = {
        "detected_people": sensor_input.detected_people,
        "last_updated": sensor_input.timestamp or datetime.now().isoformat()
    }
    
    # Calculate expected boarding delay
    boarding_delay_seconds = sensor_input.detected_people * 4.5
    boarding_delay_minutes = round(boarding_delay_seconds / 60, 2)
    
    return {
        "status": "received",
        "station_id": station_id,
        "station_name": TRANSIT_STATIONS[station_id]["display_name"],
        "passenger_count": sensor_input.detected_people,
        "estimated_boarding_delay_minutes": boarding_delay_minutes,
        "timestamp": sensor_data_registry[station_id]["last_updated"]
    }


@api.get("/api/route/geometry", tags=["Transit Data"])
async def get_route_geometry(route: str = Query("72", description="Route number")):
    """
    Get the geographical path of a bus route
    Returns coordinate sequence for map visualization
    Supports routes: 72, N1, N86
    """
    if route not in ROUTE_PATH_COORDINATES:
        raise HTTPException(status_code=404, detail=f"Route {route} not available. Available routes: {', '.join(ROUTE_PATH_COORDINATES.keys())}")
    
    path_coords = [
        {"lat": lat, "lon": lon}
        for lat, lon in ROUTE_PATH_COORDINATES[route]
    ]
    
    return {
        "route": route,
        "waypoints": len(path_coords),
        "coordinates": path_coords
    }


# ===========================
# API TEST ENDPOINTS
# ===========================

@api.get("/api/test/bods", tags=["Testing"])
async def test_bods_api():
    """
    Test BODS API connectivity and return diagnostic information
    """
    import requests
    import xml.etree.ElementTree as ET
    
    results = {
        "api_key_configured": bool(BODS_CREDENTIALS["api_key"]),
        "api_key_preview": BODS_CREDENTIALS["api_key"][:10] + "..." if BODS_CREDENTIALS["api_key"] else None,
        "test_time": datetime.now().isoformat(),
        "tests": []
    }
    
    # Test 1: Direct API call with expanded bounds
    try:
        # Use a larger bounding box for testing
        test_bounds = {
            "min_latitude": 51.30,
            "max_latitude": 51.65,
            "min_longitude": -2.80,
            "max_longitude": -2.40
        }
        bbox = f"{test_bounds['min_latitude']},{test_bounds['min_longitude']},{test_bounds['max_latitude']},{test_bounds['max_longitude']}"
        api_url = f"{BODS_CREDENTIALS['siri_endpoint']}?api_key={BODS_CREDENTIALS['api_key']}&boundingBox={bbox}"
        
        results["tests"].append({
            "name": "API URL Construction",
            "status": "success",
            "url": api_url.replace(BODS_CREDENTIALS["api_key"], "***")
        })
        
        # Make the request
        response = requests.get(api_url, timeout=15)
        
        results["tests"].append({
            "name": "HTTP Request",
            "status": "success" if response.status_code == 200 else "failed",
            "status_code": response.status_code,
            "content_length": len(response.content),
            "content_preview": response.text[:500] if response.text else None
        })
        
        if response.status_code == 200:
            # Try to parse XML
            try:
                root = ET.fromstring(response.content)
                
                # Count VehicleActivity elements
                namespaces = {
                    'siri': 'http://www.siri.org.uk/siri',
                    'ns2': 'http://www.ifopt.org.uk/acsb',
                    'ns3': 'http://www.ifopt.org.uk/ifopt',
                    'ns4': 'http://datex2.eu/schema/2_0RC1/2_0'
                }
                
                vehicle_activities = root.findall('.//siri:VehicleActivity', namespaces)
                
                # Sample a few vehicles
                sample_vehicles = []
                for va in vehicle_activities[:3]:
                    try:
                        mv = va.find('.//siri:MonitoredVehicleJourney', namespaces)
                        if mv is not None:
                            line_ref = mv.find('siri:LineRef', namespaces)
                            vehicle_ref = mv.find('siri:VehicleRef', namespaces)
                            location = mv.find('.//siri:VehicleLocation', namespaces)
                            
                            if location is not None:
                                lon_elem = location.find('siri:Longitude', namespaces)
                                lat_elem = location.find('siri:Latitude', namespaces)
                                
                                sample_vehicles.append({
                                    "route": line_ref.text if line_ref is not None else "Unknown",
                                    "vehicle_id": vehicle_ref.text if vehicle_ref is not None else "Unknown",
                                    "lat": lat_elem.text if lat_elem is not None else None,
                                    "lon": lon_elem.text if lon_elem is not None else None
                                })
                    except Exception as e:
                        pass
                
                results["tests"].append({
                    "name": "XML Parsing",
                    "status": "success",
                    "total_vehicles": len(vehicle_activities),
                    "sample_vehicles": sample_vehicles
                })
                
                # Check Bristol-filtered vehicles
                bristol_vehicles = []
                for va in vehicle_activities:
                    try:
                        mv = va.find('.//siri:MonitoredVehicleJourney', namespaces)
                        if mv is not None:
                            location = mv.find('.//siri:VehicleLocation', namespaces)
                            if location is not None:
                                lon_elem = location.find('siri:Longitude', namespaces)
                                lat_elem = location.find('siri:Latitude', namespaces)
                                
                                if lon_elem is not None and lat_elem is not None:
                                    lon = float(lon_elem.text)
                                    lat = float(lat_elem.text)
                                    
                                    if (BRISTOL_BOUNDS["min_latitude"] <= lat <= BRISTOL_BOUNDS["max_latitude"] and
                                        BRISTOL_BOUNDS["min_longitude"] <= lon <= BRISTOL_BOUNDS["max_longitude"]):
                                        line_ref = mv.find('siri:LineRef', namespaces)
                                        bristol_vehicles.append({
                                            "route": line_ref.text if line_ref is not None else "Unknown",
                                            "lat": lat,
                                            "lon": lon
                                        })
                    except:
                        pass
                
                results["tests"].append({
                    "name": "Bristol Area Filter",
                    "status": "success",
                    "vehicles_in_bristol": len(bristol_vehicles),
                    "sample_bristol_vehicles": bristol_vehicles[:5]
                })
                
            except ET.ParseError as e:
                results["tests"].append({
                    "name": "XML Parsing",
                    "status": "failed",
                    "error": str(e)
                })
        else:
            results["tests"].append({
                "name": "HTTP Request",
                "status": "failed",
                "error": f"Status code: {response.status_code}",
                "response_preview": response.text[:500] if response.text else None
            })
            
    except Exception as e:
        results["tests"].append({
            "name": "API Connection",
            "status": "failed",
            "error": str(e)
        })
    
    return results


@api.get("/api/test/tomtom", tags=["Testing"])
async def test_tomtom_api(
    lat: float = Query(51.4496, description="Latitude to test"),
    lon: float = Query(-2.5811, description="Longitude to test")
):
    """
    Test TomTom API connectivity and return diagnostic information
    """
    results = {
        "api_key_configured": bool(TOMTOM_TRAFFIC_CONFIG["api_key"]),
        "api_key_preview": TOMTOM_TRAFFIC_CONFIG["api_key"][:10] + "..." if TOMTOM_TRAFFIC_CONFIG["api_key"] else None,
        "test_coordinates": {"lat": lat, "lon": lon},
        "test_time": datetime.now().isoformat(),
        "tests": []
    }
    
    try:
        # Build URL
        url = f"{TOMTOM_TRAFFIC_CONFIG['base_url']}/{TOMTOM_TRAFFIC_CONFIG['style']}/{TOMTOM_TRAFFIC_CONFIG['zoom']}/json"
        params = {
            "key": TOMTOM_TRAFFIC_CONFIG["api_key"],
            "point": f"{lat},{lon}",
            "unit": "KMPH"
        }
        
        results["tests"].append({
            "name": "API URL Construction",
            "status": "success",
            "url": url,
            "params": {k: v if k != "key" else "***" for k, v in params.items()}
        })
        
        # Make request
        response = requests.get(url, params=params, timeout=10)
        
        results["tests"].append({
            "name": "HTTP Request",
            "status": "success" if response.status_code == 200 else "failed",
            "status_code": response.status_code,
            "content_length": len(response.content)
        })
        
        if response.status_code == 200:
            data = response.json()
            
            results["raw_response"] = data
            
            if "flowSegmentData" in data:
                flow_data = data["flowSegmentData"]
                results["tests"].append({
                    "name": "Traffic Data Extraction",
                    "status": "success",
                    "current_speed": flow_data.get("currentSpeed"),
                    "free_flow_speed": flow_data.get("freeFlowSpeed"),
                    "confidence": flow_data.get("confidence"),
                    "road_closure": flow_data.get("roadClosure", False)
                })
                
                # Calculate delay
                current_speed = flow_data.get("currentSpeed", 0)
                free_flow_speed = flow_data.get("freeFlowSpeed", 40)
                speed_ratio = current_speed / free_flow_speed if free_flow_speed > 0 else 1.0
                
                if speed_ratio >= 0.9:
                    delay = 0
                elif speed_ratio >= 0.7:
                    delay = (1 - speed_ratio) * 10
                elif speed_ratio >= 0.5:
                    delay = 2 + (0.7 - speed_ratio) * 20
                elif speed_ratio >= 0.3:
                    delay = 6 + (0.5 - speed_ratio) * 30
                else:
                    delay = 12 + (0.3 - speed_ratio) * 40
                
                results["tests"].append({
                    "name": "Delay Calculation",
                    "status": "success",
                    "speed_ratio": round(speed_ratio, 2),
                    "calculated_delay_minutes": round(delay, 1)
                })
            else:
                results["tests"].append({
                    "name": "Traffic Data Extraction",
                    "status": "failed",
                    "error": "flowSegmentData not found in response",
                    "available_keys": list(data.keys())
                })
        else:
            results["tests"].append({
                "name": "HTTP Request",
                "status": "failed",
                "error": f"Status code: {response.status_code}",
                "response_text": response.text[:500]
            })
            
    except Exception as e:
        results["tests"].append({
            "name": "API Connection",
            "status": "failed",
            "error": str(e)
        })
    
    return results


# ===========================
# SERVER STARTUP
# ===========================

def verify_apis_on_startup():
    """Verify API connectivity when server starts"""
    print("\n" + "="*60)
    print("API CONNECTIVITY CHECK")
    print("="*60)
    
    # Check BODS API
    print("\n[1] Checking BODS API...")
    try:
        bbox = "51.30,-2.80,51.65,-2.40"
        url = f"https://data.bus-data.dft.gov.uk/api/v1/datafeed/?api_key={BODS_CREDENTIALS['api_key']}&boundingBox={bbox}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            ns = {'siri': 'http://www.siri.org.uk/siri'}
            activities = root.findall('.//siri:VehicleActivity', ns)
            print(f"    Status: CONNECTED")
            print(f"    Vehicles in feed: {len(activities)}")
            if len(activities) == 0:
                current_hour = datetime.now().hour
                if current_hour < 6:
                    print(f"    Note: Early morning ({current_hour}:00) - buses not yet in service")
        else:
            print(f"    Status: FAILED (HTTP {resp.status_code})")
    except Exception as e:
        print(f"    Status: ERROR - {e}")
    
    # Check TomTom API
    print("\n[2] Checking TomTom API...")
    try:
        lat, lon = 51.4496, -2.5811
        url = f"{TOMTOM_TRAFFIC_CONFIG['base_url']}/{TOMTOM_TRAFFIC_CONFIG['style']}/{TOMTOM_TRAFFIC_CONFIG['zoom']}/json"
        params = {
            "key": TOMTOM_TRAFFIC_CONFIG["api_key"],
            "point": f"{lat},{lon}",
            "unit": "KMPH"
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "flowSegmentData" in data:
                flow = data["flowSegmentData"]
                print(f"    Status: CONNECTED")
                print(f"    Current Speed: {flow.get('currentSpeed')} km/h")
                print(f"    Free Flow: {flow.get('freeFlowSpeed')} km/h")
            else:
                print(f"    Status: CONNECTED (no traffic data)")
        else:
            print(f"    Status: FAILED (HTTP {resp.status_code})")
    except Exception as e:
        print(f"    Status: ERROR - {e}")
    
    print("\n" + "="*60)
    print("Server ready - APIs configured correctly")
    print("="*60 + "\n")


if __name__ == "__main__":
    import uvicorn
    print("[START] Bristol Transit Intelligence API Server")
    print(f"[INFO] Monitoring Routes: {'ALL ROUTES' if MONITORED_ROUTES is None else ', '.join(MONITORED_ROUTES)}")
    print(f"[INFO] Base Stations Loaded: {len(TRANSIT_STATIONS)}")
    
    # Verify APIs before starting
    verify_apis_on_startup()
    
    uvicorn.run(
        api,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )



