"""This module contains the driver assistance bot which takes the input such as driver eye
blink frequency, PERCLOS, Yawning frequency and alert the driver if he/she gets drowsy.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, StrictFloat
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


class Input(BaseModel):
    """Input data model for the driver drowsiness detection system.
    Args:
        BaseModel (pydantic.BaseModel): Provides data validation and parsing.
    """

    eye_blink_frequency: Optional[StrictFloat] = Field(
        default=None,
        description="Number of eye blinks per minute, detected from video.",
    )
    perclos: Optional[StrictFloat] = Field(
        default=None,
        description="Percentage of time eyes are closed, used to measure drowsiness.",
    )
    yawning_frequency: Optional[StrictFloat] = Field(
        default=None, description="Number of yawns per minute, detected from video."
    )


class Output(BaseModel):
    """Output data model for the driver drowsiness detection system.
    Args:
        BaseModel (pydantic.BaseModel): Provides data validation and parsing.
    """

    message: Optional[str] = Field(
        default=None,
        description="Optional text response from the model about the driver's condition or "
        "recommended actions.",
    )
    actions: Optional[List[Literal["fan", "speaker", "vibration"]]] = Field(
        default=None,
        description="List of actions to activate: 'fan' to alert the driver by blowing "
        "cool air, 'speaker' to alert, 'vibration' to warn via steering.",
    )


class Bot:
    """bot"""

    def __init__(self):
        """_summary_"""
        self.llm = OllamaLLM(model="gpt-oss:20b")
        # update the template by creating a prompt toml file
        self.prompt = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot. Your name is {name}."),
                ("human", "Hello, how are you doing?"),
                ("ai", "I'm doing well, thanks!"),
                ("human", "{user_input}"),
            ]
        )

    def invoke(self, input_data: Input) -> Output:
        """Invoke the llm
        Args:
            input_data (Input): facial features
        Returns:
            Output: llm responce and actions
        """

        bot_input = self.prompt(input_data)
        response = self.llm.invoke(bot_input)

        return response
