import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

GPT_MODEL = "gpt-4o"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_workout_program",
            "description": "Generate a workout program based on user data",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "activities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "activityName": {"type": "string"},
                                "description": {"type": "string"},
                                "day": {
                                    "type": "string",
                                    "enum": [
                                        "Sunday",
                                        "Monday",
                                        "Tuesday",
                                        "Wednesday",
                                        "Thursday",
                                        "Friday",
                                        "Saturday",
                                    ],
                                },
                                "sets": {"type": "integer"},
                                "reps": {"type": "integer"},
                            },
                        },
                    }
                },
                "required": ["activityName", "description", "day"],
            },
        },
    }
]

SYSTEM_PROMPT = """
Create a system to analyze user exercise history, and generate a customized weekly exercise plan that aims to improve the user's specific activity. 

**Tailor the plan to the sport that the user plays.** Additionally, adapt the plan according to any new input from the user.

# Steps

1. **Identify Goals**: Understand the user's goal for improvement in the specific activity (e.g., increase endurance, enhance strength, improve flexibility).
2. **Create a Weekly Plan**: 
- Formulate a personalized weekly exercise plan that aligns with the user's goals, ensuring it's balanced and progressive. 
- Always use the get_workout_program function to generate the user's workout plan. 
- Always name specific exercises the user must perform (e.g. "bicep curls" instead of "upper body strength")

** Example input: **

*Height will always be given in centimeters, and weight is always given in kilograms.*

{
    "sex": "Male",
    "weight": 77.564232,
    "height": 182.88,
    "experienceLevelMap": {
        "weight_training": "Intermediate",
        "Cycling": "No Interest",
        "Running": "Beginner"
    }
}

** Correct output: **

```JSON
{
  "activities": [
    {
      "activityName": "Barbell Bench Press",
      "description": "A compound exercise to target the chest, shoulders, and triceps.",
      "day": "Monday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Dumbbell Rows",
      "description": "An exercise to strengthen the back and improve posture.",
      "day": "Monday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Bodyweight Squats",
      "description": "A lower-body exercise for strengthening the legs and glutes.",
      "day": "Wednesday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Dumbbell Shoulder Press",
      "description": "An overhead pressing exercise to develop shoulder strength.",
      "day": "Wednesday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Incline Dumbbell Bench Press",
      "description": "Targets the upper chest and shoulders.",
      "day": "Friday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Lat Pulldowns",
      "description": "Strengthens the back and helps with pull-up progressions.",
      "day": "Friday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Light Jogging",
      "description": "A beginner-friendly cardio activity to improve running endurance.",
      "day": "Saturday",
      "sets": 1,
      "reps": 20
    },
    {
      "activityName": "Push-Ups",
      "description": "A bodyweight exercise to strengthen the chest, triceps, and core.",
      "day": "Sunday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Plank",
      "description": "A core stabilization exercise for improved overall strength.",
      "day": "Sunday",
      "sets": 4,
      "reps": 30
    }
  ]
}

```

**Example input: ** 

*Height will always be given in centimeters, and weight is always given in kilograms.*

{
    "sex": "Female",
    "weight": 63.50288,
    "height": 172.72,
    "experienceLevelMap": {
        "weight_training": "Beginner",
        "cycling": "No Interest",
        "running": "Intermediate"
    }
}

** Correct output: **

```JSON
{
  "activities": [
    {
      "activityName": "Dumbbell Goblet Squats",
      "description": "A beginner-friendly lower-body exercise to build leg strength and core stability.",
      "day": "Monday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Dumbbell Bench Press",
      "description": "A simple pressing exercise to strengthen the chest and triceps.",
      "day": "Monday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Walking Lunges",
      "description": "A functional exercise to improve lower-body strength and balance.",
      "day": "Wednesday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Dumbbell Shoulder Press",
      "description": "An overhead pressing movement to develop shoulder strength.",
      "day": "Wednesday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Romanian Deadlifts",
      "description": "A hinge movement to strengthen hamstrings and glutes.",
      "day": "Friday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Bent-Over Dumbbell Rows",
      "description": "Targets the back muscles for improved posture and strength.",
      "day": "Friday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Interval Running",
      "description": "A high-intensity running session alternating between sprints of 200 meters and recovery jogs of 400 meters.",
      "day": "Saturday",
      "sets": 6,
      "reps": 1
    },
    {
      "activityName": "Steady-State Run",
      "description": "A continuous, moderate-paced run for 5 kilometers to build aerobic endurance.",
      "day": "Sunday",
      "sets": 1,
      "reps": 1
    },
    {
      "activityName": "Bodyweight Push-Ups",
      "description": "A simple bodyweight exercise for upper-body strength.",
      "day": "Sunday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Side Plank",
      "description": "A core exercise to strengthen the obliques and improve stability.",
      "day": "Sunday",
      "sets": 4,
      "reps": 30
    }
  ]
}
```

3. **Incorporate User Input**: Adjust the weekly plan based on any new input or changes in user preferences, needs, or constraints.
"""

client = OpenAI()
app = FastAPI()


class ExperienceLevelMap(BaseModel):
    weight_training: str
    cycling: str
    running: str


class ExerciseHistory(BaseModel):
    sex: Literal["Male", "Female"]
    weight: float
    height: float
    experienceLevelMap: ExperienceLevelMap


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


@app.post("/workout")
async def read_root(exerciseHistory: ExerciseHistory):
    print(exerciseHistory)
    print(str(exerciseHistory))
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": str(exerciseHistory)})
    chat_response = chat_completion_request(
        messages,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_workout_program"}},
    )

    activities = json.loads(
        chat_response.choices[0].message.tool_calls[0].function.arguments
    )

    return activities["activities"]
