"""
ChatRF AI Tool System - Base Class

Defines the base class for AI Mode tools: drag-and-drop capabilities for
the local LLM. Drop a file implementing AITool into modules/ai_tools/,
restart AI mode, and it's automatically exposed to the LLM as a callable
function during voice conversations.
"""

import logging
from abc import ABC, abstractmethod


class AITool(ABC):
    """
    Base class for AI Mode tools.

    Attributes:
        name (str): Unique snake_case tool name. This is the function name
            sent to Ollama and the name the LLM uses when calling it.
        description (str): Plain-text description of what the tool does and
            when to use it. This is what the LLM reads to decide whether to
            call the tool.
        parameters (dict): JSON Schema describing the tool's arguments,
            following Ollama/OpenAI function-calling conventions:

            parameters = {
                "type": "object",
                "properties": {
                    "satellite_name": {
                        "type": "string",
                        "description": "Name of the satellite, e.g. ISS"
                    }
                },
                "required": ["satellite_name"]
            }

            Use {"type": "object", "properties": {}} for tools that take
            no arguments.
        enabled (bool): Whether this tool is active. Set False in
            initialize() if a dependency is missing.
        mutates_state (bool): Set True if execute() changes repeater state,
            transmits, or hits a paid/rate-limited external API. Defaults
            to False (read-only). This is purely a signal to the loader/log
            output right now.
        timeout_seconds (float): Hard cap enforced by AIToolManager around
            execute().
    """

    name = None                 # Must be set by subclass
    description = "AI tool"
    parameters = {"type": "object", "properties": {}}
    enabled = True
    mutates_state = False
    timeout_seconds = 8.0

    def __init__(self, ai):
        """
        Args:
            ai: Reference to the HamRadioAI instance (aimode.py's main
                class), gives access to self.config (AIConfig) and lets
                a tool build on shared helpers like call_with_timeout.
        """
        self.ai = ai
        self.config = ai.config
        self.logger = logging.getLogger(f"AITool.{self.name}")

        if self.name is None:
            raise ValueError(f"{self.__class__.__name__} must set a 'name' attribute")

    def initialize(self):
        """
        Called once when the tool is loaded. Override for setup, config
        reading, or checking a dependency is available. Set
        self.enabled = False here if something required is missing.
        """
        pass

    @abstractmethod
    def execute(self, **kwargs):
        """
        Run the tool and return a result.

        Args:
            **kwargs: Arguments matching the `parameters` schema, as parsed
                by Ollama from the LLM's tool call.

        Returns:
            str: A short, spoken-friendly result. This text goes straight
                 back to the LLM as the tool result and often ends up read
                 aloud over RF.

        Raises:
            Any exception - AIToolManager catches it, logs it, and feeds
            the LLM a generic failure message instead of crashing the
            AI mode session.
        """
        pass

    def cleanup(self):
        """Called on shutdown. Override to release resources."""
        pass

    def to_ollama_schema(self):
        """Build the Ollama/OpenAI-style function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# Registry populated by the loader
REGISTERED_AI_TOOLS = []


def register_ai_tool(tool_class):
    """
    Optional decorator to register a tool explicitly.
    The loader finds tools automatically by scanning modules/ai_tools/.

    Usage:
        @register_ai_tool
        class MyTool(AITool):
            ...
    """
    REGISTERED_AI_TOOLS.append(tool_class)
    return tool_class
