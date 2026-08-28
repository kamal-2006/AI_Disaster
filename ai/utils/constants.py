from __future__ import annotations

# Location constants for Erode, Tamil Nadu
ERODE_LAT = 11.3410
ERODE_LON = 77.7172
ERODE_TIMEZONE = "Asia/Kolkata"
LOCATION_NAME = "Erode, Tamil Nadu, India"

# API Endpoint
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Thresholds and Risk Levels
HIGH_RISK_THRESHOLD = 0.80
MEDIUM_RISK_THRESHOLD = 0.50

RISK_LEVEL_LOW = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH = "HIGH"

# Color Palette
COLOR_LOW_RISK = "#10B981"      # Emerald Green
COLOR_MEDIUM_RISK = "#F59E0B"   # Amber/Orange
COLOR_HIGH_RISK = "#EF4444"     # Red
COLOR_ACCENT = "#3B82F6"        # Blue
COLOR_BG_DARK = "#0F172A"       # Slate Dark

# Weather feature user-friendly names
FEATURE_DISPLAY_NAMES = {
    "max_temperature": "Max Temperature (°C)",
    "mean_temperature": "Mean Temperature (°C)",
    "min_temperature": "Min Temperature (°C)",
    "mean_relative_humidity": "Mean Humidity (%)",
    "max_relative_humidity": "Max Humidity (%)",
    "mean_wind_speed": "Wind Speed (km/h)",
    "max_wind_speed": "Max Wind Speed (km/h)",
    "max_wind_gust": "Max Wind Gust (km/h)",
    "precipitation_sum": "Precipitation (mm)",
    "mean_vpd": "Mean VPD (kPa)",
    "max_vpd": "Max VPD (kPa)",
    "mean_cloud_cover": "Cloud Cover (%)",
    "mean_dew_point": "Dew Point (°C)",
    "mean_apparent_temperature": "Mean Apparent Temp (°C)",
    "max_apparent_temperature": "Max Apparent Temp (°C)",
    "min_apparent_temperature": "Min Apparent Temp (°C)",
    "temperature_lag_1": "Temp (1 Day Ago) (°C)",
    "temperature_lag_2": "Temp (2 Days Ago) (°C)",
    "temperature_lag_3": "Temp (3 Days Ago) (°C)",
    "temperature_lag_7": "Temp (7 Days Ago) (°C)",
    "temperature_rolling_3": "3-Day Rolling Temp Avg (°C)",
    "temperature_rolling_5": "5-Day Rolling Temp Avg (°C)",
    "temperature_rolling_7": "7-Day Rolling Temp Avg (°C)",
    "relative_humidity_rolling_3": "3-Day Rolling Humidity Avg (%)",
    "relative_humidity_rolling_5": "5-Day Rolling Humidity Avg (%)",
    "relative_humidity_rolling_7": "7-Day Rolling Humidity Avg (%)",
    "apparent_temperature_rolling_3": "3-Day Rolling Apparent Temp (°C)",
    "apparent_temperature_rolling_5": "5-Day Rolling Apparent Temp (°C)",
    "apparent_temperature_rolling_7": "7-Day Rolling Apparent Temp (°C)",
    "vpd_rolling_3": "3-Day Rolling VPD Avg (kPa)",
    "vpd_rolling_5": "5-Day Rolling VPD Avg (kPa)",
    "vpd_rolling_7": "7-Day Rolling VPD Avg (kPa)",
    "year": "Year",
    "month": "Month",
    "day_of_year": "Day of Year",
}

# Preparedness Recommendations structured by persona and risk level
PREPAREDNESS_DATA = {
    "Students": {
        "LOW": [
            "Maintain baseline hydration by carrying a reusable water bottle to school.",
            "Wear light, breathable cotton clothing during outdoor playtime.",
            "Stay informed about daily heat indices through school weather boards.",
        ],
        "MEDIUM": [
            "Increase fluid intake; drink water every 30-45 minutes even if not thirsty.",
            "Avoid strenuous sports or physical games during peak sunshine (11 AM - 3 PM).",
            "Seek shaded areas or indoor climate-controlled spaces during lunch breaks.",
            "Recognize heat illness warning signs: dizziness, excessive sweating, or headache.",
        ],
        "HIGH": [
            "CRITICAL: Avoid any direct sun exposure or outdoor activities during midday hours.",
            "Sip ORS (Oral Rehydration Salts), tender coconut water, or buttermilk regularly.",
            "Immediately report symptoms of faintness, nausea, or rapid heartbeat to a teacher.",
            "Use wet cloths or cool water on wrists and forehead if feeling overheated.",
        ],
    },
    "Teachers": {
        "LOW": [
            "Encourage regular water breaks during morning and afternoon classes.",
            "Keep classroom windows open for natural ventilation when temperatures are moderate.",
        ],
        "MEDIUM": [
            "Move physical education classes indoors or schedule them before 10 AM.",
            "Monitor students for signs of heat exhaustion (flushed skin, lethargy, cramps).",
            "Ensure classroom ORS kits and clean drinking water dispensers are fully stocked.",
            "Keep blinds or curtains drawn to minimize afternoon direct solar heat gain.",
        ],
        "HIGH": [
            "Cancel all outdoor assemblies, sports activities, and field excursions.",
            "Conduct frequent hydration checks in every class period.",
            "Know the emergency protocol for heatstroke: move student to cool area, apply ice packs, call school nurse/108 ambulance.",
            "Ensure fans and ventilation systems operate at maximum efficiency.",
        ],
    },
    "Administrators": {
        "LOW": [
            "Inspect school water purification units and shade structures regularly.",
            "Review district disaster management plans for summer seasonal readiness.",
        ],
        "MEDIUM": [
            "Designate dedicated cool recovery rooms equipped with fans, ORS, and first aid.",
            "Adjust school operating hours if needed (e.g., morning shift 7:30 AM - 12:30 PM).",
            "Issue morning weather advisory advisories over school public address systems.",
            "Ensure backup power generators are ready to prevent fan/AC outages.",
        ],
        "HIGH": [
            "EMERGENCY RED ALERT: Implement modified school schedule or emergency closure directives if authorized.",
            "Deploy mobile hydration stations across campus corridors and school buses.",
            "Establish direct emergency comms with local Erode healthcare centers and 108 ambulance services.",
            "Equip security and campus transport staff with heat stress emergency kits.",
        ],
    },
    "General Public": {
        "LOW": [
            "Stay updated with Erode local weather bulletins and heat forecasts.",
            "Ensure elderly family members and pets have adequate shade and water.",
        ],
        "MEDIUM": [
            "Limit outdoor work and heavy physical exertion between 11:00 AM and 3:30 PM.",
            "Wear loose-fitting, light-colored cotton garments and UV-protective sunglasses/hats.",
            "Keep home interiors cool using wet window shades or reflective window covers.",
            "Check on vulnerable neighbors, elderly relatives, and outdoor laborers.",
        ],
        "HIGH": [
            "HIGH HEAT ALERT: Stay indoors in cool, well-ventilated or air-conditioned rooms.",
            "Avoid high-protein meals and alcohol; increase electrolyte-rich fluids (ORS, lemon water).",
            "Never leave children, elderly persons, or pets inside parked vehicles.",
            "In case of heat emergency (confusion, high body temp, dry hot skin), call 108 immediately and apply cold wet towels.",
        ],
    },
}
