from __future__ import annotations

"""
Disaster Preparedness & Emergency Knowledge Base module.
Contains verified, structured safety guidelines for major disaster categories,
emergency kit recommendations, student/school safety advice, and Erode helplines.
"""

# Erode District Official Helplines
ERODE_HELPLINES = {
    "Medical Emergency / Ambulance": "108 (TN State Ambulance Service)",
    "District Emergency Control Room": "1077 (Erode Collectorate)",
    "State Disaster Management Authority": "1070",
    "Corporation Water Supply": "1913",
    "TNEB Electricity Toll-Free": "1912",
    "Child Helpline": "1098",
    "Women Helpline": "181",
}

# General Emergency Disclaimer
DISASTER_DISCLAIMER = (
    "ℹ️ *This guidance is for educational and immediate safety reference. "
    "During active emergencies, always prioritize instructions from the Erode District Collectorate, "
    "Tamil Nadu State Disaster Management Authority (TNSDMA), and emergency responders (Call 108 / 1077).* "
    "*This AI assistant does not replace official government alerts.*"
)

# Disaster Knowledge Repository
DISASTER_KNOWLEDGE_BASE = {
    "heatwave": {
        "title": "Heatwave Safety & Extreme Heat Preparedness",
        "description": "Guidelines for staying safe during periods of abnormally high temperatures and high heat index in Erode.",
        "immediate_actions": [
            "Drink plenty of clean water, ORS (Oral Rehydration Salts), tender coconut water, or buttermilk at regular intervals even if you do not feel thirsty.",
            "Stay indoors in well-ventilated or cool shaded areas between 11:00 AM and 3:30 PM, the hottest hours of the day.",
            "Wear light-colored, loose-fitting, breathable cotton clothes and cover your head with a hat, umbrella, or cloth when stepping outside.",
            "Keep home interiors cool using damp window curtains, shades, or fan ventilation.",
            "Take frequent rest breaks in shaded areas if working outdoors, and consume electrolyte-rich drinks.",
        ],
        "what_to_avoid": [
            "Avoid strenuous physical outdoor games, sports, or heavy labor during midday peak heat.",
            "Do not consume alcohol, carbonated soft drinks, excess caffeine, or high-protein meals as they cause dehydration.",
            "NEVER leave children, elderly family members, or pets locked inside parked vehicles, even for a few minutes.",
            "Avoid direct skin exposure to intense solar radiation without protection.",
        ],
        "emergency_actions": [
            "Recognize Heat Exhaustion symptoms: heavy sweating, weakness, cold/pale skin, fast pulse, dizziness, nausea, or muscle cramps.",
            "Recognize Heatstroke symptoms (CRITICAL MEDICAL EMERGENCY): high body temperature (above 40°C/104°F), hot/red/dry or damp skin, rapid pulse, confusion, dizziness, or loss of consciousness.",
            "For Heatstroke: Move the person to a cool/air-conditioned space immediately, cool them with wet towels or ice packs on neck/armpits/groin, and CALL 108 AMBULANCE IMMEDIATELY.",
        ],
        "school_student_tips": [
            "Students should carry a full water bottle to school and drink water every 30-45 minutes.",
            "Schools should avoid outdoor sports or assemblies after 10:30 AM during heat advisories.",
            "Teachers should keep ORS packets ready in classrooms and watch for lethargic or dizzy students.",
        ],
    },
    "flood": {
        "title": "Flood Safety & Waterlogging Guidance",
        "description": "Essential precautions for heavy rainfall, river overflow (e.g. Cauvery river near Erode), and urban waterlogging.",
        "explanations": [
            "Floodwater can energize appliances and wiring, creating an electric-shock hazard even when the power appears to be off; disconnect power only from a dry, safe location.",
        ],
        "immediate_actions": [
            "Move immediately to higher ground or upper floors if rising water threatens your home.",
            "Keep emergency survival kits, important documents, medicines, and flashlights packed in waterproof containers.",
            "Listen to official Erode District Collectorate warnings and evacuate promptly if advised by local authorities.",
            "Disconnect electrical main switches and gas valves before evacuating flooded areas.",
        ],
        "what_to_avoid": [
            "Do NOT walk, swim, or drive through moving floodwaters. Just 6 inches of moving water can knock you down, and 2 feet can float a vehicle.",
            "Never touch electrical wires, fallen power poles, or submerged electrical appliances.",
            "Avoid drinking untreated floodwater or tap water after floods until declared safe by municipal authorities.",
            "Do not eat food that has come into contact with floodwaters.",
        ],
        "emergency_actions": [
            "If trapped in a building, move to the top floor or roof and signal for help using a bright cloth or flashlight.",
            "Call Erode District Control Room at 1077 or Fire & Rescue at 101 for water rescue support.",
            "Boil drinking water for at least 1 minute or use chlorine purification tablets before consumption post-flood.",
        ],
        "school_student_tips": [
            "Do not play near swollen streams, open drains, or flooded road dips.",
            "Schools in low-lying areas should suspend classes upon district flood red alerts.",
        ],
    },
    "cyclone": {
        "title": "Cyclone & High-Wind Storm Safety",
        "description": "Safety measures during cyclonic storms, severe gales, and heavy coastal/inland wind conditions.",
        "immediate_actions": [
            "Inspect and clear rooftops of loose items, tin sheets, or debris that could become flying projectiles.",
            "Secure doors, window shutters, and store adequate drinking water, non-perishable food, and emergency battery lamps.",
            "Stay indoors in the innermost room of the building away from windows and glass panels during peak winds.",
            "Keep mobile devices fully charged and store emergency numbers (1077 / 108).",
        ],
        "what_to_avoid": [
            "Do not venture outdoors during the eye of the storm (when winds temporarily calm down), as intense winds will resume suddenly from the opposite direction.",
            "Avoid taking shelter under tall trees, electric poles, tin sheds, or old dilapidated structures.",
            "Do not spread unverified rumors or panic messages on social media.",
        ],
        "emergency_actions": [
            "If wind damages the roof, seek shelter under strong tables or interior door frames.",
            "Watch out for fallen live electric lines after the storm passes and report immediately to TNEB (1912).",
            "Remain inside until local district authorities issue an official all-clear announcement.",
        ],
        "school_student_tips": [
            "Students should stay inside classroom buildings and follow teacher instructions.",
            "Keep away from glass windows and open verandas during high winds.",
        ],
    },
    "earthquake": {
        "title": "Earthquake Safety (Drop, Cover, and Hold On)",
        "description": "Life-saving emergency steps to protect yourself during ground shaking and tremors.",
        "explanations": [
            "Earthquakes happen when stress built up in the Earth's crust is suddenly released as rocks slip along a fault, sending seismic waves through the ground.",
        ],
        "immediate_actions": [
            "**DROP** down onto your hands and knees immediately to prevent being knocked over.",
            "**COVER** your head and neck under a sturdy table, desk, or furniture. If no table is nearby, cover your head with your arms against an interior wall.",
            "**HOLD ON** to your shelter until ground shaking completely stops.",
            "If outdoors, move away from buildings, streetlights, utility wires, and trees into an open area, then drop and cover.",
            "If driving, pull over safely away from overpasses, bridges, and power lines, stop, and stay inside the vehicle with seatbelts fastened.",
        ],
        "what_to_avoid": [
            "Do NOT run outdoors while the ground is actively shaking. Most injuries occur from falling bricks, glass, and debris.",
            "Do NOT use elevators or lifts during or immediately after an earthquake tremor.",
            "Do NOT stand under doorway frames unless you know they are load-bearing and reinforced.",
            "Do NOT light matches or operate electrical switches if gas leaks are suspected.",
        ],
        "emergency_actions": [
            "After shaking stops, carefully evacuate the building using stairs and gather in open assembly areas.",
            "Be prepared for aftershocks; repeat Drop, Cover, and Hold On whenever tremors recur.",
            "Check yourself and nearby people for injuries and administer basic first aid.",
        ],
        "school_student_tips": [
            "Students in classrooms should immediately crawl under desks, grip desk legs firmly, and protect their heads.",
            "Do not panic or rush to doors simultaneously to prevent stampedes.",
        ],
    },
    "landslide": {
        "title": "Landslide & Mudflow Safety",
        "description": "Precautions for hilly regions, slope instability, and sudden debris flows during heavy monsoon rains.",
        "immediate_actions": [
            "Stay alert for warning signs: cracking sounds of trees breaking, mud/water discoloration, or sudden changes in stream water level.",
            "Evacuate immediately if you hear unusual rumbling sounds or notice slope ground shifting.",
            "Move away from the direct path of mudflow or landslide towards higher stable ground.",
            "If inside a building and escape is impossible, curl into a tight ball and protect your head.",
        ],
        "what_to_avoid": [
            "Do not walk or drive across active landslide paths or mudflows.",
            "Avoid building or staying near steep cliff edges, embankments, or river bends in hilly zones.",
        ],
        "emergency_actions": [
            "Report broken utility lines or land cracks to district emergency services (1077).",
            "Help rescue trapped neighbors only if it is safe to do so without risking secondary slope collapse.",
        ],
        "school_student_tips": [
            "School buses in hilly routes should halt safely away from steep hill cuttings during heavy rainfall.",
        ],
    },
    "drought": {
        "title": "Drought Preparedness & Water Conservation",
        "description": "Long-term water management, agricultural protection, and drought mitigation strategies.",
        "immediate_actions": [
            "Practice strict water conservation: harvest rainwater, repair leaking pipes, and reuse household graywater for gardens.",
            "Store safe drinking water in clean covered containers to prevent waterborne diseases.",
            "Adopt drip irrigation and drought-resistant crops for agricultural water efficiency.",
            "Maintain proper hydration for livestock and family members.",
        ],
        "what_to_avoid": [
            "Do not waste municipal or borewell water on non-essential washing.",
            "Avoid consuming contaminated or stagnant water sources without boiling/filtration.",
        ],
        "emergency_actions": [
            "Contact Erode Corporation Water Helpline (1913) or local Panchayat for emergency tanker supply.",
        ],
        "school_student_tips": [
            "Turn off school water taps tightly after use and report leaking taps to teachers.",
        ],
    },
    "thunderstorm": {
        "title": "Thunderstorm & Heavy Downpour Safety",
        "description": "Safety guidelines during severe thunderstorms, squalls, and localized downpours.",
        "immediate_actions": [
            "Seek shelter inside a sturdy enclosed building or a hard-topped metal vehicle with windows rolled up.",
            "Unplug electronic equipment, computers, and appliances to prevent damage from power surges.",
            "Close all windows and doors securely.",
        ],
        "what_to_avoid": [
            "Do not take shelter under isolated trees, metal pole sheds, or tall structures.",
            "Avoid using corded phones or touching plumbing pipes during severe electrical storms.",
        ],
        "emergency_actions": [
            "If someone is struck by lightning, call 108 immediately and begin CPR if trained.",
        ],
        "school_student_tips": [
            "Remain inside classrooms until thunderstorm activity subsides.",
        ],
    },
    "lightning": {
        "title": "Lightning Protection & Safety Rules",
        "description": "Critical rules for surviving cloud-to-ground lightning strikes and severe electrical storms.",
        "immediate_actions": [
            "Follow the **30/30 Rule**: If time between lightning flash and thunder sound is under 30 seconds, go indoors immediately. Stay inside for 30 minutes after hearing the last thunder strike.",
            "If caught in an open field with NO shelter: crouch down low on the balls of your feet with heels touching, tuck your head, and cover your ears. Minimize ground contact.",
            "If inside a metal vehicle, keep windows closed and avoid touching metal parts.",
        ],
        "what_to_avoid": [
            "NEVER stand under tall isolated trees, open verandas, or metal fence lines.",
            "Do NOT stay in open water, ponds, or swimming pools during lightning.",
            "Do NOT lie flat on the ground; crouching minimizes ground current path.",
            "Avoid holding metal objects such as umbrellas with metal tips, golf clubs, or bicycles.",
        ],
        "emergency_actions": [
            "Lightning victims carry NO electrical charge and are safe to touch immediately.",
            "Check for pulse/breathing, perform CPR if necessary, and call 108 Ambulance immediately.",
        ],
        "school_student_tips": [
            "Stop all outdoor playground activities immediately at the first sound of thunder.",
        ],
    },
    "tsunami": {
        "title": "Tsunami Warning & Coastal Evacuation",
        "description": "Emergency evacuation steps for coastal tsunami alerts and underwater earthquake activity.",
        "immediate_actions": [
            "If you feel a strong earthquake near coastal areas or notice sea water suddenly receding rapidly, evacuate inland to higher ground (at least 30 meters above sea level or 2 km inland) IMMEDIATELY.",
            "Do not wait for an official tsunami warning if natural warning signs (ground shaking, sea receding, loud roar) occur.",
            "Stay on high ground; tsunamis are a series of multiple waves over several hours.",
        ],
        "what_to_avoid": [
            "NEVER go down to the beach to observe a receding sea or tsunami wave.",
            "Do not return to low-lying coastal zones until official clearance is declared by TNSDMA.",
        ],
        "emergency_actions": [
            "Follow coastal evacuation route markers and assist elderly neighbors.",
        ],
        "school_student_tips": [
            "Coastal schools should follow established tsunami evacuation hill routes calmly.",
        ],
    },
    "emergency_kit": {
        "title": "Disaster Emergency Survival Kit Essentials",
        "description": "Recommended items to assemble in your family or school emergency preparedness bag (Go-Bag).",
        "immediate_actions": [
            "**Water**: At least 3-5 liters of clean drinking water per person for 3 days.",
            "**Food**: Non-perishable food items (dry snacks, biscuits, nuts, energy bars, canned goods).",
            "**First Aid Kit**: Bandages, antiseptic solution (Dettol/Betadine), burn ointment, ORS packets, paracetamol, cotton, scissor, prescription medicines.",
            "**Tools & Lighting**: LED Flashlight with extra batteries, whistle (to signal for help), multi-tool pocket knife, emergency blanket.",
            "**Communication**: Battery-operated or hand-crank radio, fully charged power bank, mobile phone.",
            "**Sanitation & Documents**: Hand sanitizer, wet wipes, soap, waterproof pouch containing copies of ID proofs, insurance, land records, and emergency cash.",
        ],
        "what_to_avoid": [
            "Do not store expired medicines, leaky batteries, or perishable cooked food in your emergency kit.",
            "Avoid packing excessively heavy non-essential items that hinder rapid evacuation.",
        ],
        "emergency_actions": [
            "Check and refresh emergency kit items every 6 months (water, batteries, food expiration dates).",
        ],
        "school_student_tips": [
            "Students should keep a mini emergency card in their school bag with home address, parent contacts, and medical blood group.",
        ],
    },
    "general_prep": {
        "title": "General Disaster Preparedness & Risk Reduction",
        "description": "Fundamental emergency planning rules for homes, schools, and communities.",
        "immediate_actions": [
            "**Create a Family Disaster Plan**: Agree on an emergency meeting spot outside your home and designate an out-of-district contact person.",
            "**Keep Emergency Numbers Saved**: Save 108 (Ambulance), 1077 (District Control Room), 101 (Fire), 100 (Police), 1912 (Electricity).",
            "**Conduct Regular Drills**: Practice earthquake Drop-Cover-Hold drills and evacuation routes at home and in schools.",
            "**Stay Informed**: Monitor local news bulletins, IMD weather advisories, and SafeGraph AI risk warnings.",
        ],
        "what_to_avoid": [
            "Do not ignore official weather advisories or warning sirens.",
            "Do not spread fake rumors or panic messages.",
        ],
        "emergency_actions": [
            "Help vulnerable community members, including senior citizens, pregnant women, and persons with disabilities during emergencies.",
        ],
        "school_student_tips": [
            "Participate actively in school mock disaster safety drills.",
        ],
    },
}

# Categorization mapping helper
KEYWORD_CATEGORY_MAP = {
    "heatwave": ["heatwave", "heat wave", "extreme heat", "sunstroke", "heat stroke", "heat exhaustion", "heat stress", "dehydration", "hot day", "temperature high", "going outside", "outside today", "precautions today"],
    "flood": ["flood", "flooding", "waterlog", "water logging", "overflow", "river cauvery", "heavy rain", "inundation"],
    "cyclone": ["cyclone", "storm", "gale", "hurricane", "typhoon", "high wind", "sea storm"],
    "earthquake": ["earthquake", "tremor", "ground shaking", "seismic", "aftershock", "quake"],
    "landslide": ["landslide", "mudslide", "mud flow", "slope failure", "rockfall", "hill collapse"],
    "drought": ["drought", "water scarcity", "dry spell", "water shortage", "no rain"],
    "thunderstorm": ["thunderstorm", "thunder", "squall", "heavy downpour"],
    "lightning": ["lightning", "thunderbolt", "electrical storm", "lightening"],
    "tsunami": ["tsunami", "sea wave", "tidal wave", "coastal wave"],
    "emergency_kit": ["emergency kit", "go bag", "survival kit", "first aid kit", "what items", "what to pack", "supplies"],
    "general_prep": ["prepare", "preparedness", "before a disaster", "after a disaster", "early warning", "school prep", "schools prepare", "disaster plan", "safety drill"],
}
