import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  Box, Drawer, CircularProgress, IconButton, Typography, TextField,
  Paper, Chip, Avatar, Stack, Divider, Card, CardContent, List, ListItem, ListItemText
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import CloseIcon from '@mui/icons-material/Close';
import MenuIcon from '@mui/icons-material/Menu';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import DirectionsBusIcon from '@mui/icons-material/DirectionsBus';
import InfoIcon from '@mui/icons-material/Info';
import GroupIcon from '@mui/icons-material/Group';
import SpeedIcon from '@mui/icons-material/Speed';
import PlaceIcon from '@mui/icons-material/Place';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle } from 'react-leaflet';
import L from 'leaflet';

import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

// Configure default Leaflet marker
const defaultMarker = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = defaultMarker;

// ===========================
// CUSTOM MAP ICONS
// ===========================

// User location marker (pulsing blue dot)
const userLocationMarker = L.divIcon({
  html: `<div style="
    width: 18px; height: 18px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: 4px solid rgba(255,255,255,0.9);
    border-radius: 50%;
    box-shadow: 0 0 0 4px rgba(102,126,234,0.3);
    animation: pulse 2s infinite;
  "></div>
  <style>
    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 0 4px rgba(102,126,234,0.3); }
      50% { box-shadow: 0 0 0 10px rgba(102,126,234,0.1); }
    }
  </style>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
  className: 'user-location-marker'
});

// Bus stop marker (colored by selection state)
const createStopMarker = (isActive) => L.divIcon({
  html: `<div style="
    width: ${isActive ? 48 : 40}px;
    height: ${isActive ? 48 : 40}px;
    background: ${isActive ? 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)' : 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'};
    border: 4px solid white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
  ">
    <span style="font-size: ${isActive ? 24 : 20}px;">🚏</span>
  </div>`,
  iconSize: [isActive ? 48 : 40, isActive ? 48 : 40],
  iconAnchor: [isActive ? 24 : 20, isActive ? 24 : 20],
  className: 'transit-stop-marker'
});

// Bus vehicle marker (shows route number)
const createVehicleMarker = (routeNumber, colorCode) => L.divIcon({
  html: `<div style="
    width: 56px;
    height: 56px;
    background: ${colorCode};
    border: 5px solid white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 900;
    font-size: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    z-index: 10000;
  ">${routeNumber}</div>`,
  iconSize: [56, 56],
  iconAnchor: [28, 28],
  className: 'vehicle-marker'
});

// ===========================
// UTILITY FUNCTIONS
// ===========================

const ROUTE_COLOR_PALETTE = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B4D9', '#A8E6CF'
];

const getColorForRoute = (routeNum) => {
  let hashValue = 0;
  const routeStr = String(routeNum);
  for (let i = 0; i < routeStr.length; i++) {
    hashValue = routeStr.charCodeAt(i) + ((hashValue << 5) - hashValue);
  }
  return ROUTE_COLOR_PALETTE[Math.abs(hashValue) % ROUTE_COLOR_PALETTE.length];
};

const formatETA = (minutes) => {
  if (minutes <= 0.5) return 'Arriving';
  if (minutes < 1) return '< 1 min';
  return `${Math.round(minutes)} min`;
};

const formatDistance = (kilometers) => {
  if (kilometers < 1) return `${Math.round(kilometers * 1000)} m`;
  return `${kilometers.toFixed(1)} km`;
};

// ===========================
// MAP COMPONENT
// ===========================

function TransitMapView({ 
  userPosition, 
  transitStations, 
  activeStation, 
  trackedVehicle, 
  displayMode, 
  onStationSelect,
  stationVehicles,
  onVehicleSelect
}) {
  const mapInstance = useRef(null);
  const isInitialized = useRef(false);
  
  // Center map on user location on first render
  useEffect(() => {
    if (!isInitialized.current && mapInstance.current && userPosition) {
      mapInstance.current.setView(userPosition, 13);
      isInitialized.current = true;
    }
  }, [userPosition]);
  
  // Follow tracked vehicle
  useEffect(() => {
    if (trackedVehicle && mapInstance.current) {
      mapInstance.current.setView(
        [trackedVehicle.current_position.lat, trackedVehicle.current_position.lon],
        15,
        { animate: true }
      );
    }
  }, [trackedVehicle?.vehicle_id]);
  
  if (!userPosition) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  // Extract vehicle trail coordinates
  const vehicleTrail = trackedVehicle?.position_trail?.map(point => [point.lat, point.lon]) || [];
  
  return (
    <MapContainer
      center={userPosition}
      zoom={13}
      style={{ height: '100%', width: '100%' }}
      ref={mapInstance}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://openstreetmap.org">OpenStreetMap</a>'
      />
      
      {/* User Location Marker */}
      {userPosition && (
        <Marker position={userPosition} icon={userLocationMarker}>
          <Popup>
            <Typography variant="body2" fontWeight={700}>Your Location</Typography>
          </Popup>
        </Marker>
      )}
      
      {/* Transit Station Markers */}
      {displayMode === 'stations' && transitStations.map((station) => (
        <Marker
          key={station.id}
          position={[station.latitude, station.longitude]}
          icon={createStopMarker(activeStation?.id === station.id)}
          eventHandlers={{
            click: () => onStationSelect(station)
          }}
        >
          <Popup>
            <Box sx={{ minWidth: 200 }}>
              <Typography variant="subtitle1" fontWeight={700}>{station.name}</Typography>
              <Typography variant="caption" color="text.secondary">
                Route {station.route} - Stop #{station.sequence}
              </Typography>
            </Box>
          </Popup>
        </Marker>
      ))}
      
      {/* Approaching Vehicles */}
      {displayMode === 'vehicles' && stationVehicles.map((vehicle) => (
        <Marker
          key={vehicle.vehicle_id}
          position={[vehicle.current_position.lat, vehicle.current_position.lon]}
          icon={createVehicleMarker(vehicle.route, getColorForRoute(vehicle.route))}
          eventHandlers={{
            click: () => onVehicleSelect(vehicle)
          }}
        >
          <Popup>
            <Box sx={{ minWidth: 180 }}>
              <Typography variant="h6" fontWeight={900}>Route {vehicle.route}</Typography>
              <Typography variant="body2">→ {vehicle.destination}</Typography>
              <Divider sx={{ my: 1 }} />
              <Typography variant="caption">Distance: {formatDistance(vehicle.distance_km)}</Typography>
              <br />
              <Typography variant="caption">ETA: {formatETA(vehicle.eta_minutes)}</Typography>
            </Box>
          </Popup>
        </Marker>
      ))}
      
      {/* Tracked Vehicle with Trail */}
      {trackedVehicle && (
        <>
          <Marker
            position={[trackedVehicle.current_position.lat, trackedVehicle.current_position.lon]}
            icon={createVehicleMarker(trackedVehicle.route, getColorForRoute(trackedVehicle.route))}
            zIndexOffset={5000}
          >
            <Popup>
              <Box sx={{ minWidth: 200 }}>
                <Typography variant="h5" fontWeight={900}>Route {trackedVehicle.route}</Typography>
                <Typography variant="body1">→ {trackedVehicle.destination}</Typography>
                <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                  Operator: {trackedVehicle.operator}
                </Typography>
              </Box>
            </Popup>
          </Marker>
          
          {/* GPS Trail Visualization */}
          {vehicleTrail.length > 1 && (
            <>
              <Polyline
                positions={vehicleTrail}
                pathOptions={{
                  color: 'white',
                  weight: 12,
                  opacity: 0.7,
                  lineCap: 'round',
                  lineJoin: 'round'
                }}
              />
              <Polyline
                positions={vehicleTrail}
                pathOptions={{
                  color: getColorForRoute(trackedVehicle.route),
                  weight: 7,
                  opacity: 1,
                  lineCap: 'round',
                  lineJoin: 'round',
                  dashArray: '10, 5'
                }}
              />
            </>
          )}
        </>
      )}
    </MapContainer>
  );
}

// ===========================
// MAIN APP COMPONENT
// ===========================

function BristolTransitApp() {
  const [displayMode, setDisplayMode] = useState('stations');
  const [userPosition, setUserPosition] = useState(null);
  const [activeStation, setActiveStation] = useState(null);
  const [stationVehicles, setStationVehicles] = useState([]);
  const [trackedVehicle, setTrackedVehicle] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastRefreshTime, setLastRefreshTime] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [availableStations, setAvailableStations] = useState([]);
  
  const SERVER_BASE_URL = "http://127.0.0.1:8000/api";
  
  // Get user location on mount
  useEffect(() => {
    if(navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserPosition([position.coords.latitude, position.coords.longitude]);
        },
        () => {
          // Fallback to Bristol city center
          setUserPosition([51.4545, -2.5879]);
        }
      );
    } else {
      setUserPosition([51.4545, -2.5879]);
    }
  }, []);
  
  // Load all stations on mount
  useEffect(() => {
    const loadStations = async () => {
      try {
        const response = await axios.get(`${SERVER_BASE_URL}/stations`, {
          params: { route: '72' }
        });
        setAvailableStations(response.data.stations || []);
      } catch (error) {
        console.error("Failed to load stations:", error);
      }
    };
    loadStations();
  }, []);
  
  // Fetch vehicles approaching selected station
  const fetchApproachingVehicles = useCallback(async (station) => {
    if (!station || !userPosition) return;
    
    setIsLoading(true);
    try {
      const response = await axios.get(
        `${SERVER_BASE_URL}/stations/${station.id}/approaching`
      );
      
      setStationVehicles(response.data.vehicles || []);
      setLastRefreshTime(new Date());
    } catch (error) {
      console.error("Failed to fetch vehicles:", error);
      setStationVehicles([]);
    } finally {
      setIsLoading(false);
    }
  }, [userPosition]);
  
  // Auto-refresh vehicle data
  useEffect(() => {
    if (!activeStation || displayMode !=='vehicles') return;
    
    fetchApproachingVehicles(activeStation);
    const refreshInterval = setInterval(() => {
      fetchApproachingVehicles(activeStation);
    }, 10000); // Refresh every 10 seconds
    
    return () => clearInterval(refreshInterval);
  }, [activeStation, displayMode, fetchApproachingVehicles]);
  
  // Filter stations by search term
  const filteredStations = availableStations.filter(station =>
    station.name.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  const handleStationSelection = (station) => {
    setActiveStation(station);
    setTrackedVehicle(null);
    setDisplayMode('vehicles');
    setSidebarOpen(true);
    fetchApproachingVehicles(station);
  };
  
  const handleVehicleSelection = (vehicle) => {
    setTrackedVehicle(vehicle);
    setDisplayMode('tracking');
  };
  
  const handleBackToStations = () => {
    setActiveStation(null);
    setStationVehicles([]);
    setTrackedVehicle(null);
    setDisplayMode('stations');
    setSidebarOpen(false);
  };
  
  const handleBackToVehicles = () => {
    setTrackedVehicle(null);
    setDisplayMode('vehicles');
  };
  
  return (
    <Box sx={{ display: 'flex', height: '100vh', width: '100%', overflow: 'hidden' }}>
      <style>{`
        .user-location-marker, .transit-stop-marker, .vehicle-marker {
          background: transparent !important;
          border: none !important;
        }
      `}</style>
      
      {/* Main Map Area */}
      <Box sx={{ flex: 1, position: 'relative' }}>
        <TransitMapView
          userPosition={userPosition}
          transitStations={availableStations}
          activeStation={activeStation}
          trackedVehicle={trackedVehicle}
          displayMode={displayMode}
          onStationSelect={handleStationSelection}
          stationVehicles={stationVehicles}
          onVehicleSelect={handleVehicleSelection}
        />
        
        {/* Map Overlay - Header */}
        {displayMode === 'stations' && (
          <Box sx={{ 
            position: 'absolute', 
            top: 20, 
            left: 20, 
            right: 20, 
            zIndex: 1000 
          }}>
            <Paper 
              elevation={4} 
              sx={{ 
                p: 2.5, 
                borderRadius: 3,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white'
              }}
            >
              <Typography variant="h5" fontWeight={900}>Bristol Transit Intelligence</Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                Route 72: Temple Meads ↔ UWE Frenchay
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                Select a stop to view real-time bus arrivals
              </Typography>
            </Paper>
          </Box>
        )}
        
        {/* Search Panel */}
        {displayMode === 'stations' && (
          <Box sx={{ 
            position: 'absolute', 
            top: 140, 
            left: 20, 
            right: 20, 
            zIndex: 1000 
          }}>
            <Paper elevation={3} sx={{ borderRadius: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', p: 1.5 }}>
                <SearchIcon sx={{ color: '#9ca3af', mr: 1 }} />
                <TextField
                  fullWidth
                  placeholder="Search stops..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  variant="standard"
                  InputProps={{ disableUnderline: true }}
                />
                {searchTerm && (
                  <IconButton size="small" onClick={() => setSearchTerm('')}>
                    <CloseIcon fontSize="small" />
                  </IconButton>
                )}
              </Box>
              
              {searchTerm && filteredStations.length > 0 && (
                <Box sx={{ maxHeight: 300, overflow: 'auto', borderTop: '1px solid #e5e7eb' }}>
                  {filteredStations.map((station) => (
                    <Box
                      key={station.id}
                      onClick={() => {
                        handleStationSelection(station);
                        setSearchTerm('');
                      }}
                      sx={{
                        p: 2,
                        borderBottom: '1px solid #f3f4f6',
                        cursor: 'pointer',
                        '&:hover': { bgcolor: '#f9fafb' }
                      }}
                    >
                      <Typography fontWeight={600}>{station.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Stop #{station.sequence} • Route {station.route}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </Paper>
          </Box>
        )}
        
        {/* Menu Button for Mobile */}
        {displayMode !== 'stations' && (
          <IconButton
            onClick={() => setSidebarOpen(true)}
            sx={{
              position: 'absolute',
              top: 20,
              left: 20,
              zIndex: 1000,
              bgcolor: 'white',
              boxShadow: 3,
              '&:hover': { bgcolor: '#f5f5f5' }
            }}
          >
            <MenuIcon />
          </IconButton>
        )}
      </Box>
      
      {/* Side Panel */}
      <Drawer
        anchor="right"
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        variant={displayMode === 'stations' ? 'temporary' : 'persistent'}
        sx={{
          '& .MuiDrawer-paper': {
            width: { xs: '100%', sm: 400 },
            boxSizing: 'border-box'
          }
        }}
      >
        <Box sx={{ height: '100%', overflow: 'auto' }}>
          {/* Station Info View */}
          {displayMode === 'vehicles' && activeStation && (
            <Box sx={{ p: 3 }}>
              {/* Header */}
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <IconButton onClick={handleBackToStations} sx={{ mr: 1 }}>
                  <CloseIcon />
                </IconButton>
                <Typography variant="h6" fontWeight={700}>Station Details</Typography>
              </Box>
              
              {/* Station Card */}
              <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', color: 'white' }}>
                <CardContent>
                  <Typography variant="h5" fontWeight={900}>{activeStation.name}</Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Route {activeStation.route} - Stop #{activeStation.sequence}
                  </Typography>
                  <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                    <Chip
                      icon={<PlaceIcon style={{ color: 'white' }} />}
                      label={`${activeStation.latitude.toFixed(4)}, ${activeStation.longitude.toFixed(4)}`}
                      size="small"
                      sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                    />
                  </Box>
                </CardContent>
              </Card>
              
              {/* Vehicles List */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1.5 }}>
                  Approaching Buses ({stationVehicles.length})
                </Typography>
                
                {isLoading && stationVehicles.length === 0 ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : stationVehicles.length === 0 ? (
                  <Paper sx={{ p: 3, textAlign: 'center', bgcolor: '#f9fafb' }}>
                    <DirectionsBusIcon sx={{ fontSize: 48, color: '#9ca3af', mb: 1 }} />
                    <Typography color="text.secondary">
                      No buses nearby at the moment
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                      Real-time data from BODS API
                    </Typography>
                  </Paper>
                ) : (
                  <Stack spacing={1.5}>
                    {stationVehicles.map((vehicle) => (
                      <Card
                        key={vehicle.vehicle_id}
                        onClick={() => handleVehicleSelection(vehicle)}
                        sx={{
                          cursor: 'pointer',
                          transition: 'all 0.3s',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: 4
                          }
                        }}
                      >
                        <CardContent sx={{ p: 2 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Avatar
                              sx={{
                                bgcolor: getColorForRoute(vehicle.route),
                                fontWeight: 900,
                                fontSize: 20
                              }}
                            >
                              {vehicle.route}
                            </Avatar>
                            <Box sx={{ flex: 1 }}>
                              <Typography fontWeight={700}>→ {vehicle.destination}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                {formatDistance(vehicle.distance_km)} away
                              </Typography>
                            </Box>
                            <Box sx={{ textAlign: 'right' }}>
                              <Typography variant="h6" fontWeight={900} color="primary">
                                {formatETA(vehicle.eta_minutes)}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                arrival
                              </Typography>
                            </Box>
                          </Box>
                          
                          {/* Additional Info */}
                          <Box sx={{ display: 'flex', gap: 1, mt: 1.5, flexWrap: 'wrap' }}>
                            {/* Passenger count indicator */}
                            <Chip
                              icon={<GroupIcon />}
                              label={vehicle.crowd_delay > 0 ? `${vehicle.crowd_delay}min delay` : 'No passengers'}
                              size="small"
                              color={vehicle.crowd_delay > 0 ? "info" : "default"}
                              variant="outlined"
                            />
                            
                            {/* Traffic indicator */}
                            {vehicle.traffic_delay > 0 ? (
                              <Chip
                                icon={<SpeedIcon />}
                                label={`Traffic: +${vehicle.traffic_delay}min`}
                                size="small"
                                color="warning"
                                variant="outlined"
                              />
                            ) : (
                              <Chip
                                icon={<SpeedIcon />}
                                label="No traffic"
                                size="small"
                                color="default"
                                variant="outlined"
                              />
                            )}
                            
                            {/* Data source indicator */}
                            <Chip
                              icon={<span>{vehicle.data_source === 'BODS_SIRI_VM' ? '✓' : '⚠'}</span>}
                              label={vehicle.data_source === 'BODS_SIRI_VM' ? 'Live GPS' : vehicle.data_source}
                              size="small"
                              color={vehicle.data_source === 'BODS_SIRI_VM' ? 'success' : 'warning'}
                              variant="outlined"
                            />
                          </Box>
                        </CardContent>
                      </Card>
                    ))}
                  </Stack>
                )}
              </Box>
              
              {/* Last Update */}
              {lastRefreshTime && (
                <Typography variant="caption" color="text.secondary" align="center" display="block" sx={{ mt: 2 }}>
                  Updated: {lastRefreshTime.toLocaleTimeString()}
                </Typography>
              )}
            </Box>
          )}
          
          {/* Vehicle Tracking View */}
          {displayMode === 'tracking' && trackedVehicle && (
            <Box sx={{ p: 3 }}>
              {/* Header */}
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <IconButton onClick={handleBackToVehicles} sx={{ mr: 1 }}>
                  <CloseIcon />
                </IconButton>
                <Typography variant="h6" fontWeight={700}>Live Vehicle Tracking</Typography>
              </Box>
              
              {/* Vehicle Info Card */}
              <Card sx={{ mb: 3, background: `linear-gradient(135deg, ${getColorForRoute(trackedVehicle.route)} 0%, ${getColorForRoute(trackedVehicle.route)}dd 100%)`, color: 'white' }}>
                <CardContent>
                  <Typography variant="h3" fontWeight={900} sx={{ mb: 1 }}>
                    {trackedVehicle.route}
                  </Typography>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    → {trackedVehicle.destination}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                    <Box>
                      <Typography variant="caption" sx={{ opacity: 0.8 }}>Distance</Typography>
                      <Typography fontWeight={700} variant="h6">
                        {formatDistance(trackedVehicle.distance_km)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ opacity: 0.8 }}>ETA</Typography>
                      <Typography fontWeight={700} variant="h6">
                        {formatETA(trackedVehicle.eta_minutes)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ opacity: 0.8 }}>Operator</Typography>
                      <Typography fontWeight={700}>
                        {trackedVehicle.operator}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
              
              {/* ETA Breakdown */}
              <Paper sx={{ p: 2.5, mb: 3, bgcolor: '#f8f9fa' }}>
                <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 2 }}>
                  <InfoIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                  ETA Calculation
                </Typography>
                
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="Base travel time"
                      secondary={`${((trackedVehicle.distance_km / 25) * 60).toFixed(1)} min`}
                    />
                  </ListItem>
                  {trackedVehicle.traffic_delay > 0 && (
                    <ListItem>
                      <ListItemText
                        primary="Traffic delay"
                        secondary={`+${trackedVehicle.traffic_delay} min`}
                        primaryTypographyProps={{ color: 'warning.main' }}
                      />
                    </ListItem>
                  )}
                  {trackedVehicle.crowd_delay > 0 && (
                    <ListItem>
                      <ListItemText
                        primary="Boarding delay (crowd)"
                        secondary={`+${trackedVehicle.crowd_delay} min`}
                        primaryTypographyProps={{ color: 'info.main' }}
                      />
                    </ListItem>
                  )}
                  <Divider sx={{ my: 1 }} />
                  <ListItem>
                    <ListItemText
                      primary="Total ETA"
                      secondary={formatETA(trackedVehicle.eta_minutes)}
                      primaryTypographyProps={{ fontWeight: 700 }}
                    />
                  </ListItem>
                </List>
              </Paper>
              
              {/* Data Sources */}
              <Paper sx={{ p: 2.5, bgcolor: '#e3f2fd' }}>
                <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1.5 }}>
                  🔄 Real-Time Data Sources
                </Typography>
                
                {/* Data Source 1: Bus GPS */}
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    borderRadius: '50%', 
                    bgcolor: trackedVehicle.data_source === 'BODS_SIRI_VM' ? '#4caf50' : '#f44336',
                    mr: 1 
                  }} />
                  <Typography variant="body2">
                    <strong>1. Bus GPS:</strong> BODS API (Live)
                  </Typography>
                </Box>
                
                {/* Data Source 2: Traffic */}
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    borderRadius: '50%', 
                    bgcolor: trackedVehicle.traffic_data_source?.includes('TomTom') ? '#4caf50' : '#ff9800',
                    mr: 1 
                  }} />
                  <Typography variant="body2">
                    <strong>2. Traffic:</strong> {trackedVehicle.traffic_data_source || 'Not available'}
                  </Typography>
                </Box>
                
                {/* Data Source 3: Passenger Count */}
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Box sx={{ 
                    width: 10, 
                    height: 10, 
                    borderRadius: '50%', 
                    bgcolor: trackedVehicle.crowd_delay > 0 ? '#4caf50' : '#9e9e9e',
                    mr: 1 
                  }} />
                  <Typography variant="body2">
                    <strong>3. Passengers:</strong> CV Sensor (YOLOv8) {trackedVehicle.crowd_delay > 0 ? '- Detected' : '- No data'}
                  </Typography>
                </Box>
                
                <Divider sx={{ my: 1 }} />
                
                {/* ML Model Info */}
                <Typography variant="caption" color="text.secondary">
                  Predictions use XGBoost ML model combining all three data sources
                </Typography>
              </Paper>
              
              {/* Traffic Details (if available) */}
              {trackedVehicle.traffic_details && (
                <Paper sx={{ p: 2.5, mt: 2, bgcolor: '#fff3e0' }}>
                  <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1.5 }}>
                    🚦 Traffic Details (TomTom)
                  </Typography>
                  <Typography variant="body2">
                    Current Speed: {trackedVehicle.traffic_details.current_speed} km/h
                  </Typography>
                  <Typography variant="body2">
                    Free Flow: {trackedVehicle.traffic_details.free_flow_speed} km/h
                  </Typography>
                  <Typography variant="body2">
                    Congestion: {Math.round((1 - trackedVehicle.traffic_details.speed_ratio) * 100)}%
                  </Typography>
                </Paper>
              )}
            </Box>
          )}
        </Box>
      </Drawer>
    </Box>
  );
}

export default BristolTransitApp;
