"""
ChatRF AI Tool Manager

Discovers and loads AI tools from modules/ai_tools/. Tools are
exposed to Ollama during AI mode conversations so the LLM can
call them as functions to extend its capabilities.

Owned by HamRadioAI (aimode.py).
"""

import importlib
import inspect
import logging
from pathlib import Path

from modules.ai_tools.base import AITool

logger = logging.getLogger(__name__)


class AIToolManager:
    """
    Manages all AI Mode tools: discovery, schema generation, and
    timeout-guarded execution.
    """

    def __init__(self, ai):
        self.ai = ai
        self.tools = {}   # {tool_name: tool_instance}

    def load_all_tools(self, tools_dir="modules/ai_tools"):
        """
        Discover and load all AI tools from the tools directory.

        Args:
            tools_dir: Path to the ai_tools directory
        """
        logger.info("--------------------------")
        logger.info("Loading AI tools...")

        tools_path = Path(tools_dir)
        if not tools_path.exists():
            logger.warning(f"AI tools directory not found: {tools_dir}")
            return

        module_root = tools_dir.replace("/", ".")

        for tool_file in tools_path.glob("*.py"):
            if tool_file.name.startswith("_") or tool_file.name in ("base.py", "ai_tool_manager.py"):
                continue

            module_name = f"{module_root}.{tool_file.stem}"
            self._load_tool_module(module_name)

        logger.info(f"AI tool loading complete: {len(self.tools)} tool(s) active")
        for name in self.tools:
            logger.info(f"  - {name}")

    def _load_tool_module(self, module_name):
        """Load a single tool file and register any AITool subclasses in it"""
        try:
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module_name:
                    continue

                if issubclass(obj, AITool) and obj is not AITool:
                    self._register_tool(obj)

        except Exception as e:
            logger.error(f"Failed to load AI tool file {module_name}: {e}", exc_info=True)

    def _register_tool(self, tool_class):
        """Instantiate, initialize, and register a single tool"""
        try:
            instance = tool_class(self.ai)

            if not instance.enabled:
                logger.info(f"Skipping disabled AI tool: {instance.name}")
                return

            instance.initialize()

            if not instance.enabled:
                # initialize() may have disabled itself (e.g. missing dependency)
                logger.info(f"AI tool disabled itself during initialize(): {instance.name}")
                return

            if instance.name in self.tools:
                logger.warning(
                    f"AI tool name '{instance.name}' conflict: "
                    f"keeping the first one loaded, skipping {tool_class}"
                )
                return

            self.tools[instance.name] = instance

            tag = "MUTATES STATE" if instance.mutates_state else "read-only"
            logger.info(f"Loaded AI tool: {instance.name} ({tag})")

        except Exception as e:
            logger.error(f"Error registering AI tool {tool_class}: {e}", exc_info=True)

    def get_schemas(self):
        """
        Build the list of Ollama-format tool schemas for every loaded tool.
        Pass this straight into ollama_client.chat(..., tools=...).

        Returns:
            list[dict], or [] if no tools are loaded
        """
        return [tool.to_ollama_schema() for tool in self.tools.values()]

    def execute_tool(self, name, arguments):
        """
        Execute a tool by name with a hard timeout, catching any error.
        Always returns a string suitable to feed back to the LLM.

        Args:
            name: Tool name as called by the LLM
            arguments: dict of parsed arguments from the tool call

        Returns:
            str: Tool result, or a short failure message on error/timeout/unknown tool
        """
        tool = self.tools.get(name)
        if tool is None:
            logger.warning(f"LLM requested unknown tool: {name}")
            return f"Tool '{name}' is not available."

        arguments = arguments or {}

        # aimode.py imports AIToolManager, so AIToolManager can't import
        # aimode.py at the top level.
        from aimode import call_with_timeout

        try:
            result = call_with_timeout(
                tool.execute,
                kwargs=arguments,
                timeout=tool.timeout_seconds,
            )
            return str(result) if result else "The tool ran but returned no result."

        except TimeoutError:
            logger.error(f"AI tool '{name}' timed out after {tool.timeout_seconds}s")
            return f"The {name} tool took too long to respond."

        except Exception as e:
            logger.error(f"AI tool '{name}' raised an error: {e}", exc_info=True)
            return f"The {name} tool ran into an error."

    def shutdown_all(self):
        """Cleanup all loaded tools"""
        for tool in self.tools.values():
            try:
                tool.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up AI tool {tool.name}: {e}")
