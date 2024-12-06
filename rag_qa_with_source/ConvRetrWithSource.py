from typing import Callable, List, Dict, Any, Optional, Union, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from genflow import CustomComponent
from genflow.field_typing import (
    BaseLLM,
    BaseLanguageModel,
    BaseRetriever,
    Chain,
    BaseMemory,
    BasePromptTemplate,
    Document,
)
import inspect
from langchain.schema import BaseMessage, PromptValue
from langchain.utils.input import get_colored_text


class CustomConversationalRetrievalWithSourceChain(CustomComponent):
    """
    A custom component for implementing a conversational retrieval QA chain.
    """

    display_name: str = "ConversationalRetrievalQAWithSource"
    description: str = "A conversational QA chain with retrieval and memory."
    documentation: str = (
        "https://docs.aiplanet.com/components/chains#conversationalretrievalqa"
    )
    beta: bool = True

    def build_config(self):
        return {
            "llm": {"display_name": "LLM", "required": True},
            "prompt": {"display_name": "Prompt"},
            "memory": {"display_name": "Memory"},
            "retriever": {"display_name": "Retriever", "required": True},
            "code": {"show": True},
        }

    @property
    def custom_conversation_chain(self):
        class CustomConversationRetrievalChain(Chain):
            """Chain for having a conversation based on retrieved documents."""

            llm: BaseLanguageModel
            memory: BaseMemory
            prompt: BasePromptTemplate

            retriever: BaseRetriever
            """Retriever to use to fetch documents."""
            output_key: str = "answer"
            """The output key to return the final answer of this chain in."""

            get_chat_history: Optional[
                Callable[[List[Union[Tuple[str, str], BaseMessage]]], str]
            ] = None
            """An optional function to get a string of the chat history."""

            _ROLE_MAP = {"human": "Human: ", "ai": "Assistant: "}

            @property
            def input_keys(self) -> List[str]:
                """Input keys."""
                return [
                    key
                    for key in self.prompt.input_variables
                    if key not in ["answer", "context"]
                ]

            @property
            def output_keys(self) -> List[str]:
                """Return the output keys.

                :meta private:
                """
                return [self.output_key]

            def _get_chat_history(
                    self, chat_history: List[Union[Tuple[str, str], BaseMessage]]
            ) -> str:
                buffer = ""
                for dialogue_turn in chat_history:
                    if isinstance(dialogue_turn, BaseMessage):
                        role_prefix = {
                            "human": "HumanMessage: ",
                            "ai": "AIMessage: ",
                        }.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
                        buffer += f"\n{role_prefix}{dialogue_turn.content}"
                    elif isinstance(dialogue_turn, tuple):
                        human = "Human: " + dialogue_turn[0]
                        ai = "Assistant: " + dialogue_turn[1]
                        buffer += "\n" + "\n".join([human, ai])
                    else:
                        raise ValueError(
                            f"Unsupported chat history format: {type(dialogue_turn)}."
                            f" Full chat history: {chat_history} "
                        )
                return buffer

            def prep_prompts(
                    self,
                    input_list: List[Dict[str, Any]],
                    run_manager: Optional[CallbackManagerForChainRun] = None,
            ) -> Tuple[List[PromptValue], Optional[List[str]]]:
                """Prepare prompts from inputs."""
                stop = None
                if len(input_list) == 0:
                    return [], stop
                if "stop" in input_list[0]:
                    stop = input_list[0]["stop"]
                prompts = []
                for inputs in input_list:
                    selected_inputs = {
                        k: inputs[k] for k in self.prompt.input_variables
                    }
                    prompt = self.prompt.format_prompt(**selected_inputs)
                    _colored_text = get_colored_text(prompt.to_string(), "green")
                    _text = "Prompt after formatting:\n" + _colored_text
                    if run_manager:
                        run_manager.on_text(_text, end="\n", verbose=self.verbose)
                    if "stop" in inputs and inputs["stop"] != stop:
                        raise ValueError(
                            "If `stop` is present in any inputs, should be present in all."
                        )
                    prompts.append(prompt)
                return prompts, stop

            async def aprep_prompts(
                    self,
                    input_list: List[Dict[str, Any]],
                    run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
            ) -> Tuple[List[PromptValue], Optional[List[str]]]:
                """Prepare prompts from inputs."""
                stop = None
                if len(input_list) == 0:
                    return [], stop
                if "stop" in input_list[0]:
                    stop = input_list[0]["stop"]
                prompts = []
                for inputs in input_list:
                    selected_inputs = {
                        k: inputs[k] for k in self.prompt.input_variables
                    }
                    prompt = self.prompt.format_prompt(**selected_inputs)
                    _colored_text = get_colored_text(prompt.to_string(), "green")
                    _text = "Prompt after formatting:\n" + _colored_text
                    if run_manager:
                        await run_manager.on_text(_text, end="\n", verbose=self.verbose)
                    if "stop" in inputs and inputs["stop"] != stop:
                        raise ValueError(
                            "If `stop` is present in any inputs, should be present in all."
                        )
                    prompts.append(prompt)
                return prompts, stop

            def _process_documents(self, docs):
                """Process each document to append formatted metadata to page_content."""
                for doc in docs:
                    # Generate metadata string, skipping empty keys
                    metadata_string = "\n".join(
                        f"{key}: {str(value)}"
                        for key, value in doc.metadata.items()
                        if key.strip()
                    )
                    doc.page_content += f"\nMetadata:\n{metadata_string}"
                return docs


            def _get_docs(
                    self,
                    question: str,
                    inputs: Dict[str, Any],
                    *,
                    run_manager: CallbackManagerForChainRun,
            ) -> List[Document]:
                """Get docs."""
                docs = self.retriever.get_relevant_documents(
                    question, callbacks=run_manager.get_child()
                )
                return self._process_documents(docs)

            async def _aget_docs(
                    self,
                    question: str,
                    inputs: Dict[str, Any],
                    *,
                    run_manager: AsyncCallbackManagerForChainRun,
            ) -> List[Document]:
                """Get docs."""
                docs = await self.retriever.aget_relevant_documents(
                    question, callbacks=run_manager.get_child()
                )
                return self._process_documents(docs)

            def _call(
                    self,
                    inputs: Dict[str, Any],
                    run_manager: Optional[CallbackManagerForChainRun] = None,
            ) -> Dict[str, Any]:
                _run_manager = (
                        run_manager or CallbackManagerForChainRun.get_noop_manager()
                )
                question = inputs["question"]

                chat_history_str = self._get_chat_history(inputs["chat_history"])

                accepts_run_manager = (
                        "run_manager" in inspect.signature(self._get_docs).parameters
                )

                if accepts_run_manager:
                    docs = self._get_docs(question, inputs, run_manager=_run_manager)
                else:
                    docs = self._get_docs(question, inputs)  # type: ignore[call-arg]

                new_inputs = inputs.copy()
                new_inputs["chat_history"] = chat_history_str
                new_inputs["context"] = docs

                prompts, stop = self.prep_prompts([new_inputs], run_manager=run_manager)

                response = self.llm.generate_prompt(
                    prompts,
                    stop,
                    callbacks=run_manager.get_child() if run_manager else None,
                )
                generated_text = response.generations[0][0].text

                output: Dict[str, Any] = {self.output_key: generated_text}
                return output

            async def _acall(
                    self,
                    inputs: Dict[str, Any],
                    run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
            ) -> Dict[str, Any]:
                _run_manager = (
                        run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
                )
                question = inputs["question"]

                chat_history_str = self._get_chat_history(inputs["chat_history"])

                accepts_run_manager = (
                        "run_manager" in inspect.signature(self._aget_docs).parameters
                )

                if accepts_run_manager:
                    docs = await self._aget_docs(
                        question, inputs, run_manager=_run_manager
                    )
                else:
                    docs = await self._aget_docs(question, inputs)  # type: ignore[call-arg]

                new_inputs = inputs.copy()
                new_inputs["chat_history"] = chat_history_str
                new_inputs["context"] = docs

                prompts, stop = await self.aprep_prompts(
                    [new_inputs], run_manager=run_manager
                )

                response = await self.llm.agenerate_prompt(
                    prompts,
                    stop,
                    callbacks=run_manager.get_child() if run_manager else None,
                )
                generated_text = response.generations[0][0].text

                output: Dict[str, Any] = {self.output_key: generated_text}
                return output

        return CustomConversationRetrievalChain

    def build(
            self,
            llm: BaseLanguageModel,
            retriever: BaseRetriever,
            prompt: Optional[BasePromptTemplate] = None,
            memory: BaseMemory = None,
    ) -> Chain:
        if prompt is None:
            prompt_template = """ You're a friendly and helpful AI chatbot.  Answer the following question in a friendly manner, 
            ensuring you utilize the provided context and chat history to inform your response.

                                {context}

                                Chat History: {chat_history}

                                Question: {question}
                                Answer:
                            """

            QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(
                prompt_template,
            )
        else:
            QA_CHAIN_PROMPT = prompt

        return self.custom_conversation_chain(
            prompt=QA_CHAIN_PROMPT,
            llm=llm,
            memory=memory,
            retriever=retriever,
        )