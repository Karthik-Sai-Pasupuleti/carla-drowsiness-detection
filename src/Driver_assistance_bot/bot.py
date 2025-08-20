"""This module contains the driver assistance bot which takes the input such as driver eye
blink frequency, PERCLOS, Yawning frequency and alert the driver if he/she gets drowsy.
"""

import json
from pprint import pprint
from typing import List, Literal, Optional
import toml
from pydantic import BaseModel, Field, StrictFloat
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def load_json(file_path: str) -> dict:
    """Load and parse json file

    Args:
        file_path (str): path to json file

    Returns:
        dict: parsed data
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_toml(file_path: str) -> dict:
    """Load and parse toml file

    Args:
        file_path (str):  path to toml file

    Returns:
        dict: parsed data
    """
    data = toml.load(file_path)
    return data


class Input(BaseModel):
    """
    Input data model for the driver drowsiness detection system.

    Args:
        BaseModel (pydantic.BaseModel): Provides data validation and parsing.
    """

    perclos: Optional[StrictFloat] = Field(
        default=None,
        description="Percentage of time eyes are closed, used to measure drowsiness.",
    )
    blink_rate: Optional[StrictFloat] = Field(
        default=None,
        description="Number of eye blinks per minute, detected from video.",
    )
    yawn_freq: Optional[StrictFloat] = Field(
        default=None, description="Number of yawns per minute, detected from video."
    )
    sdlp: Optional[StrictFloat] = Field(
        default=None, description="Standard deviation of lane position in meters."
    )
    steering_entropy: Optional[StrictFloat] = Field(
        default=None, description="Unpredictability measure of steering movements."
    )
    steering_reversal_rate: Optional[StrictFloat] = Field(
        default=None,
        description="Number of significant steering direction changes per minute.",
    )


class Output(BaseModel):
    """
    Output data model for the driver drowsiness detection system.

    Args:
        BaseModel (pydantic.BaseModel): Provides data validation and parsing.
    """

    drowsiness_level: Literal["low", "medium", "high"] = Field(
        description="Detected drowsiness risk level."
    )

    reasoning: str = Field(
        description="Brief explanation based on input metrics that led to the decision."
    )

    actions: List[Literal["speaker", "fan", "steering_vibration"]] = Field(
        description="List of actions to trigger in response to detected drowsiness."
    )


class Bot:
    """Driver drowsiness detection bot"""

    def __init__(self, model_id: str, prompts: dict, schema: dict):
        llm = ChatOllama(model=model_id)

        # Load prompts from TOML
        system_prompt = prompts["bot_prompt"]["SYSTEM"]
        user_prompt = prompts["bot_prompt"]["USER"]

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

        self.str_llm = llm.with_structured_output(schema)

    def invoke(self, input_data: Input) -> Output:
        """
        Invoke the LLM
        Args:
            input_data (Input): Driver monitoring and vehicle metrics
        Returns:
            Output: LLM response parsed into Output model
        """
        formatted_messages = self.prompt.invoke({"drowsiness_metrics": input_data})

        raw_response = self.str_llm.invoke(formatted_messages)
        output = Output.model_validate(raw_response)

        return output


# gpt-oss:20b

if __name__ == "__main__":
    prompts_ = load_toml(r"src\Driver_assistance_bot\configs\prompt.toml")
    schema_ = load_json(r"src\Driver_assistance_bot\configs\schema.json")
    bot_ = Bot("llama3.1:8b", prompts_, schema_)
    llm_input = {
        "perclos": 30,
        "blink_rate": 60,
        "yawn_freq": 20,
        "sdlp": 0.1,
        "steering_entropy": 0.6,
        "steering_reversal_rate": 10,
    }
    result = bot_.invoke(llm_input)
    pprint(result.dict())
