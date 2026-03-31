from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import requests
from call_groq import call_groq

app = FastAPI(title="ReAct Hospitality Advisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Real weather tool using Open-Meteo (no API key)
def get_weather(destination: str, month: str) -> str:
    """
    Fetch real weather for a destination using Open-Meteo.
    Returns a human-readable summary.
    """
    # Geocode the destination name to coordinates
    geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={destination}&count=1&language=en&format=json"
    try:
        geo_resp = requests.get(geocode_url, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data.get('results'):
            return f"Could not find coordinates for {destination}. Please try a well‑known city name."

        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        city_name = geo_data['results'][0]['name']

        # Get current weather (or you could use a forecast for the specific month)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
        weather_resp = requests.get(weather_url, timeout=10)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        current = weather_data.get('current_weather', {})
        temp = current.get('temperature')
        if temp is None:
            return f"Could not retrieve temperature for {city_name}."

        return f"The current weather in {city_name} (representative for {month}) is {temp}°C."
    except requests.exceptions.RequestException as e:
        return f"Weather service error: {str(e)}"

class ReActRequest(BaseModel):
    destination: str
    month: str

class ReActResponse(BaseModel):
    final_answer: str
    steps: list
    elapsed_time: float

@app.post("/react", response_model=ReActResponse)
async def react_endpoint(request: ReActRequest):
    start_time = time.time()
    steps = []

    # Build the ReAct prompt – instruct the model to use the get_weather tool
    system_prompt = f"""You are a helpful travel advisor. The user is planning a trip to {request.destination} in {request.month}.
You have a tool: get_weather(destination, month). Use it to check the weather before giving packing advice.
Follow this ReAct format exactly:

Thought: [your reasoning about what to do next]
Action: get_weather({request.destination}, {request.month})
Observation: [the system will provide the weather result]

After you have the observation, continue reasoning and provide a final answer.

Final Answer: [your packing advice based on the weather, e.g., "It's cold, bring a sweater" or "Hot and humid, pack summer clothes and an umbrella."]
"""

    # First call: get the action (should be get_weather)
    response = call_groq(system_prompt, node_name="REACT_INIT")
    steps.append({"step": "Thought & Action", "output": response})

    # Extract action – if it contains the tool call, execute it
    if "Action: get_weather" in response:
        # Actually call the real weather tool
        weather_info = get_weather(request.destination, request.month)
        observation = f"Observation: {weather_info}"
        steps.append({"step": "Tool Result", "output": observation})

        # Second call: continue the conversation with the observation
        conversation = system_prompt + "\n" + response + "\n" + observation
        final = call_groq(conversation, node_name="REACT_FINAL")
        steps.append({"step": "Final Answer", "output": final})
        final_answer = final
    else:
        # If the model didn't call the tool (rare), use its response as final
        final_answer = response
        steps.append({"step": "Fallback Answer", "output": response})

    elapsed = time.time() - start_time
    return ReActResponse(final_answer=final_answer, steps=steps, elapsed_time=elapsed)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)