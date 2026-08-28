"""Drop-in adapters for the pipelines people already have.

Each adapter is a thin wrapper: it imports its framework lazily, so installing
turnchunk never drags in LangChain or LlamaIndex, and importing an adapter you
don't use costs nothing.

    from turnchunk.integrations.langchain import TurnChunkSplitter
    from turnchunk.integrations.llamaindex import TurnChunkNodeParser
    from turnchunk.integrations.chonkie import ConversationChunker
"""

__all__ = ["chonkie", "langchain", "llamaindex"]
