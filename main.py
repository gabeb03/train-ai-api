import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

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

1. **Analyze Exercise History**: The exercise history is in this format: 

```JSON
{
    "athlete_type": <an enum of either "Runner", "Cyclist", "Swimmer", "Other">,
    "data": [
        {
           "avg_split": <some value of type float>,
           "avg_distance": <some value of type float>
        }
    ],
    "description": <a description of the user of type string>
}
```

If the `athelete_type` is "Other", the description will **always** provide context about the user. Also, the exercise history will not be given.

The `data` attribute will not be included in the exercise history object if the user is of type "Other".

2. **Identify Goals**: Understand the user's goal for improvement in the specific activity (e.g., increase endurance, enhance strength, improve flexibility).
3. **Create a Weekly Plan**: 
Formulate a personalized weekly exercise plan that aligns with the user's goals, ensuring it's balanced and progressive. 
Always use the get_workout_program function to generate the user's workout plan. 
Always name specific exercises the user must perform (e.g. "bicep curls" instead of "upper body strength")

**Example input: ** athlete_type='Other' description='Male. 170LBS. 6 feet tall. Wants to gain 15 LBS of muscle mass. Max bench press: 170LBS. Max dead lift: 100LBS.'

** Correct output: **

```JSON
{
  "activities": [
    {
      "activityName": "Bench Press",
      "description": "Upper Body Strength",
      "day": "Monday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Arnolds Press",
      "description": "Upper Body Strength",
      "day": "Monday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Bent-over Rows",
      "description": "Upper Body Strength",
      "day": "Monday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Tricep Pushdowns",
      "description": "Upper Body Strength",
      "day": "Monday",
      "sets": 3,
      "reps": 15
    },
    {
      "activityName": "Dumbbell Bicep Curls",
      "description": "Upper Body Strength",
      "day": "Monday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Deadlift",
      "description": "Lower Body Strength",
      "day": "Tuesday",
      "sets": 4,
      "reps": 10
    },
    {
      "activityName": "Squats",
      "description": "Lower Body Strength",
      "day": "Tuesday",
      "sets": 4,
      "reps": 12
    },
    {
      "activityName": "Leg Press",
      "description": "Lower Body Strength",
      "day": "Tuesday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Calf Raises",
      "description": "Lower Body Strength",
      "day": "Tuesday",
      "sets": 3,
      "reps": 15
    },
    {
      "activityName": "Incline Dumbbell Press",
      "description": "Hypertrophy Focus",
      "day": "Thursday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Lateral Raises",
      "description": "Hypertrophy Focus",
      "day": "Thursday",
      "sets": 3,
      "reps": 15
    },
    {
      "activityName": "Seated Rows",
      "description": "Hypertrophy Focus",
      "day": "Thursday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Skull Crushers",
      "description": "Hypertrophy Focus",
      "day": "Thursday",
      "sets": 3,
      "reps": 15
    },
    {
      "activityName": "Hammer Curls",
      "description": "Hypertrophy Focus",
      "day": "Thursday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Romanian Deadlifts",
      "description": "Lower Body and Core",
      "day": "Friday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Lunges",
      "description": "Lower Body and Core",
      "day": "Friday",
      "sets": 3,
      "reps": 12
    },
    {
      "activityName": "Plank",
      "description": "Lower Body and Core",
      "day": "Friday",
      "sets": 3,
      "reps": 1
    },
    {
      "activityName": "Abdominal Crunches",
      "description": "Lower Body and Core",
      "day": "Friday",
      "sets": 3,
      "reps": 20
    },
    {
      "activityName": "Kettlebell Exercises",
      "description": "Functional and Flexibility",
      "day": "Sunday",
      "sets": 3,
      "reps": 15
    },
    {
      "activityName": "Stretching",
      "description": "Functional and Flexibility",
      "day": "Sunday",
      "sets": 1,
      "reps": 30
    }
  ]
}
```

**Example input: ** athlete_type='Runner' data=Data(avg_split=5.25, avg_distance=10.0) description='Female. 150LBS. 5ft, 6in.'

** Correct output: **

```JSON
{
    "activities": [
        {
            "activityName": "Distance Run",
            "description": "A steady-paced run to build endurance.",
            "day": "Monday",
            "sets": 1,
            "reps": 1
        },
        {
            "activityName": "Interval Training",
            "description": "Short bursts of high-intensity running with rest in between.",
            "day": "Tuesday",
            "sets": 6,
            "reps": 400
        },
        {
            "activityName": "Recovery Run",
            "description": "Light run to aid recovery and build stamina.",
            "day": "Wednesday",
            "sets": 1,
            "reps": 1
        },
        {
            "activityName": "Tempo Run",
            "description": "Run at a challenging but sustainable pace to enhance threshold.",
            "day": "Thursday",
            "sets": 1,
            "reps": 1
        },
        {
            "activityName": "Cross-Training",
            "description": "Incorporate other activities like cycling or swimming.",
            "day": "Friday",
            "sets": 3,
            "reps": 10
        },
        {
            "activityName": "Long Run",
            "description": "Run at a conversational pace to build endurance.",
            "day": "Saturday",
            "sets": 1,
            "reps": 1
        },
        {
            "activityName": "Rest Day",
            "description": "Day off from running to recover.",
            "day": "Sunday",
            "sets": 0,
            "reps": 0
        }
    ]
}
```

4. **Incorporate User Input**: Adjust the weekly plan based on any new input or changes in user preferences, needs, or constraints.
"""

client = OpenAI()
app = FastAPI()


class Data(BaseModel):
    avg_split: float
    avg_distance: float


class ExerciseHistory(BaseModel):
    athlete_type: str
    data: Union[Data, None] = None
    description: str


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
