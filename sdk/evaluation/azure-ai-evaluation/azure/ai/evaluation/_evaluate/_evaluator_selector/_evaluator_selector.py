# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import json
import asyncio
from typing import Dict, Optional, Union

from azure.core.credentials import TokenCredential
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase


class EvaluatorSelector(PromptyEvaluatorBase[Dict[str, str]]):
    """
    Evaluator Selector that determines which evaluators to use based on conversation history.

    This class analyzes the conversation history between a user and an agent to decide which 
    evaluators should be applied for content safety and quality assessment. It does not perform
    evaluation itself, but rather selects appropriate evaluators based on the content.

    The implementation uses an LLM with the associated prompty file for intelligent
    evaluator selection. LLM-based operation is required - if the LLM is not available
    or properly configured, initialization will fail with a RuntimeError.

    The available evaluators it can select from include:
    - HateUnfairnessEvaluator: For hate speech and unfair representations
    - SexualEvaluator: For sexual content
    - ViolenceEvaluator: For violent content  
    - SelfHarmEvaluator: For self-harm content
    - ProtectedMaterialEvaluator: For copyrighted material
    - IndirectAttackEvaluator: For indirect jailbreak attacks
    - CodeVulnerabilityEvaluator: For code security vulnerabilities

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param credential: Azure credential for authentication.
    :type credential: Optional[~azure.core.credentials.TokenCredential]
    :raises RuntimeError: If LLM-based evaluation is not available or fails to initialize.

    .. admonition:: Example:

        .. code-block:: python

            from azure.ai.evaluation import AzureOpenAIModelConfiguration
            from azure.ai.evaluation._evaluate._evaluator_selector import EvaluatorSelector

            model_config = AzureOpenAIModelConfiguration(
                azure_endpoint="https://your-endpoint.openai.azure.com/",
                api_key="your-api-key",
                azure_deployment="your-deployment-name"
            )

            selector = EvaluatorSelector(model_config=model_config)
            
            conversation_history = '''
            User: Can you help me write Python code for a web application?
            Assistant: I'll help you create a basic web application using Flask...
            '''
            
            result = selector(conversation_history=conversation_history)
            print(result)  # {"CodeVulnerabilityEvaluator": "The conversation includes code snippets..."}

    .. note::
        
        This selector requires the associated _evaluator_selector.prompty file and proper
        LLM configuration to function. It will raise RuntimeError if the required dependencies
        are not available or if the LLM cannot be properly initialized.
    """

    _PROMPTY_FILE = "_evaluator_selector.prompty"
    _RESULT_KEY = "evaluator_selection"

    def __init__(self, model_config, *, credential: Optional[TokenCredential] = None, **kwargs):
        """
        Initialize the EvaluatorSelector.

        :param model_config: Configuration for the model to use.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :param credential: Azure credential for authentication.
        :type credential: Optional[TokenCredential]
        :raises RuntimeError: If LLM-based evaluation is not available due to missing dependencies.
        :raises ValueError: If model configuration is invalid.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        
        super().__init__(
            result_key=self._RESULT_KEY,
            prompty_file=prompty_path,
            model_config=model_config,
            credential=credential,
            **kwargs,
        )

    def __call__(self, *, conversation_history: str) -> Dict[str, str]:
        """Select evaluators based on conversation history.

        :keyword conversation_history: The conversation history to analyze for evaluator selection.
        :paramtype conversation_history: str
        :return: Dictionary with selected evaluators and their selection reasons.
        :rtype: Dict[str, str]
        """
        return self.select_evaluators(conversation_history)

    async def _do_eval(self, eval_input: Dict) -> Dict[str, str]:
        """
        Async evaluation method following the evaluator pattern.

        :param eval_input: The input to the evaluator containing conversation_history
        :type eval_input: Dict
        :return: Dictionary mapping evaluator names to selection reasons.
        :rtype: Dict[str, str]
        :raises RuntimeError: If LLM call fails or returns invalid response.
        """
        conversation_history = eval_input.get("conversation_history", "")
        if not conversation_history:
            raise RuntimeError("conversation_history is required in eval_input")
        
        try:
            # Call the prompty with the conversation history
            result = await self._flow(timeout=self._LLM_CALL_TIMEOUT, conversation_history=conversation_history)
            
            # Parse the JSON response from the LLM
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        return parsed_result
                    else:
                        raise RuntimeError(
                            f"LLM returned invalid response format. Expected dict, got {type(parsed_result)}."
                        )
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"LLM returned invalid JSON response: {result[:200]}..."
                    ) from e
            elif isinstance(result, dict):
                return result
            else:
                raise RuntimeError(
                    f"LLM returned unexpected response type. Expected str or dict, got {type(result)}."
                )
                
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise  # Re-raise RuntimeError as-is
            raise RuntimeError(
                f"LLM-based evaluator selection failed: {str(e)}"
            ) from e

    def select_evaluators(self, conversation_history: str) -> Dict[str, str]:
        """
        Select evaluators using LLM-based analysis of conversation history.

        :param conversation_history: The conversation history to analyze.
        :type conversation_history: str
        :return: Dictionary mapping evaluator names to selection reasons.
        :rtype: Dict[str, str]
        :raises RuntimeError: If LLM-based evaluation fails.
        """
        eval_input = {"conversation_history": conversation_history}
        
        # Check if we're in a nested event loop environment first
        try:
            asyncio.get_running_loop()
            # If we're here, there's a running loop, so use the thread-based approach directly
            import concurrent.futures
            
            def run_in_thread():
                # Create a new event loop in the thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Create and run the async function entirely within this thread
                    async def thread_async_eval():
                        return await self._do_eval(eval_input)
                    
                    return loop.run_until_complete(thread_async_eval())
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)  # Clean up thread-local event loop
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
                
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(self._do_eval(eval_input))

    def get_available_evaluators(self) -> Dict[str, str]:
        """
        Get a list of all available evaluators that can be selected.

        :return: Dictionary mapping evaluator names to their descriptions.
        :rtype: Dict[str, str]
        """
        return {
            "HateUnfairnessEvaluator": "Measures hate speech and unfair representations based on race, ethnicity, gender, etc.",
            "SexualEvaluator": "Measures sexual content including anatomical references, romantic relationships, etc.", 
            "ViolenceEvaluator": "Measures violent content including physical actions intended to harm or kill.",
            "SelfHarmEvaluator": "Measures content related to self-harm or suicide.",
            "ProtectedMaterialEvaluator": "Measures presence of copyrighted material like song lyrics, recipes, articles.",
            "IndirectAttackEvaluator": "Measures indirect jailbreak attempts and cross-domain prompt injections.",
            "CodeVulnerabilityEvaluator": "Measures security vulnerabilities in code across multiple programming languages."
        }
