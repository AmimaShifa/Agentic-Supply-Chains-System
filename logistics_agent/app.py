# Standard Library Imports
import asyncio
import pprint
import datetime
import json
import operator
import voyageai
import pickle
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from types import TracebackType
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
from typing_extensions import Self

# Third-Party Library Imports
import pandas as pd
import pymongo
from pymongo.operations import SearchIndexModel
import streamlit as st
import anthropic
from tqdm import tqdm
from langchain_voyageai import VoyageAIEmbeddings
from langchain.agents import tool
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
from langchain_anthropic import ChatAnthropic
from motor.motor_asyncio import AsyncIOMotorClient
# LangChain Core Imports
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

# LangGraph Imports
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

# MongoDB Driver Imports
from motor.motor_asyncio import AsyncIOMotorClient


MONGO_URI = st.secrets["MONGO_URI"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
VOYAGE_API_KEY = st.secrets["VOYAGE_API_KEY"]

client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

VOYAGEAI_EMBEDDING_MODEL = "voyage-finance-2"
VOYAGEAI_EMBEDDING_MODEL_DIMENSION = 1024

embedding_model = VoyageAIEmbeddings(
    voyage_api_key=VOYAGE_API_KEY,
    model=VOYAGEAI_EMBEDDING_MODEL
)

st.title(" 🌐 AI AGENTIC SUPPLY CHAIN SYSTEM 📦 ")
st.write(
    "Enhanced Contract and Supply Chain Management for International Shipping"
)

def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.rag.supply_chain.python"
    )

    # Validate the connection
    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        # Connection successful
        print("Connection to MongoDB successful")
        return client
    print("Connection to MongoDB failed")
    return None

mongo_client = get_mongo_client(MONGO_URI)
DB_NAME = "supply_chain"
COLLECTION_NAME = "shipping_data"

# Create or get the database
db = mongo_client[DB_NAME]

# Create or get the collections
collection = db[COLLECTION_NAME]

# The field containing the text embeddings on each document within the shipping_data collection
embedding_field_name = "embedding"
# MongoDB Atlas Vector Search index name
vector_search_index_name = "vector_index"


#MongoDB LangGraph Checkpointer
class JsonPlusSerializerCompat(JsonPlusSerializer):
    def loads(self, data: bytes) -> Any:
        if data.startswith(b"\x80") and data.endswith(b"."):
            return pickle.loads(data)
        return super().loads(data)


class MongoDBSaver(AbstractContextManager, BaseCheckpointSaver):
    serde = JsonPlusSerializerCompat()

    client: AsyncIOMotorClient
    db_name: str
    collection_name: str

    def __init__(
        self,
        client: AsyncIOMotorClient,
        db_name: str,
        collection_name: str,
        *,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.client = client
        self.db_name = db_name
        self.collection_name = collection_name
        self.collection = client[db_name][collection_name]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return True

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if config["configurable"].get("thread_ts"):
            query = {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": config["configurable"]["thread_ts"],
            }
        else:
            query = {"thread_id": config["configurable"]["thread_id"]}

        doc = await self.collection.find_one(query, sort=[("thread_ts", -1)])
        if doc:
            return CheckpointTuple(
                config,
                self.serde.loads(doc["checkpoint"]),
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "thread_ts": doc["parent_ts"],
                        }
                    }
                    if doc.get("parent_ts")
                    else None
                ),
            )
        return None

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        query = {}
        if config is not None:
            query["thread_id"] = config["configurable"]["thread_id"]
        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value
        if before is not None:
            query["thread_ts"] = {"$lt": before["configurable"]["thread_ts"]}

        cursor = self.collection.find(query).sort("thread_ts", -1)
        if limit:
            cursor = cursor.limit(limit)

        async for doc in cursor:
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "thread_ts": doc["thread_ts"],
                    }
                },
                self.serde.loads(doc["checkpoint"]),
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "thread_ts": doc["parent_ts"],
                        }
                    }
                    if doc.get("parent_ts")
                    else None
                ),
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[dict[str, Union[str, float, int]]],
    ) -> RunnableConfig:
        doc = {
            "thread_id": config["configurable"]["thread_id"],
            "thread_ts": checkpoint["id"],
            "checkpoint": self.serde.dumps(checkpoint),
            "metadata": self.serde.dumps(metadata),
        }
        if config["configurable"].get("thread_ts"):
            doc["parent_ts"] = config["configurable"]["thread_ts"]
        await self.collection.insert_one(doc)
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["id"],
            }
        }

    # Implement synchronous methods as well for compatibility
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError("Use aget_tuple for asynchronous operations")

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        raise NotImplementedError("Use alist for asynchronous operations")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        raise NotImplementedError("Use aput for asynchronous operations")

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Asynchronously store intermediate writes linked to a checkpoint."""
        docs = []
        for channel, value in writes:
            doc = {
                "thread_id": config["configurable"]["thread_id"],
                "task_id": task_id,
                "channel": channel,
                "value": self.serde.dumps(value),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            docs.append(doc)

        if docs:
            await self.collection.insert_many(docs)


#Create Collection Search Tool
# Create search index
def create_collection_search_index(collection, index_definition, index_name):
    """
    Create a search index for a MongoDB Atlas collection.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary defining the index mappings
    index_name: String name for the index

    Returns:
    str: Result of the index creation operation
    """

    try:
        search_index_model = SearchIndexModel(
            definition=index_definition, name=index_name
        )

        result = collection.create_search_index(model=search_index_model)
        print(f"Search index '{index_name}' created successfully")
        return result
    except Exception as e:
        print(f"Error creating search index: {e!s}")
        return None


def print_collection_search_indexes(collection):
    """
    Print all search indexes for a given collection.

    Args:
    collection: MongoDB collection object
    """
    print(f"\nSearch indexes for collection '{collection.name}':")
    for index in collection.list_search_indexes():
        print(f"Index: {index['name']}")

shipping_data_text_index_definition = {
    "mappings": {
        "dynamic": True,
        "fields": {
            "Goods Description": {"type": "string"},
            "Keywords": {"type": "string"},
            "Jurisdiction": {"type": "string"},
            "Destination": {"type": "string"},
            "Origin": {"type": "string"},
            "Shipper": {"type": "string"},
            "Receiver": {"type": "string"},
        },
    }
}

# Create Search Index for corpus collection
## Create Search Index for corpus collection
TEXT_SEARCH_INDEX = "text_search_index"

#create_collection_search_index(
 #   db[COLLECTION_NAME], shipping_data_text_index_definition, TEXT_SEARCH_INDEX
#)
# Vector Stores Intialisation
vector_store_supply_chain_records = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embedding_model,
    index_name=vector_search_index_name,
    text_key="Keywords",
)

hybrid_search = MongoDBAtlasHybridSearchRetriever(
    vectorstore=vector_store_supply_chain_records,
    search_index_name=TEXT_SEARCH_INDEX,
    top_k=10,
)

# Let's test that hybrid search works
query = "What are the customs requirements for toys shipped to Canada?"
hybrid_search_result = hybrid_search.get_relevant_documents(query)
pprint.pprint(hybrid_search_result)

@tool
def supply_chain_hybrid_search_tool(query: str):
    """
    Perform a hybrid (vector + full-text) search on inventory, contracts, shipping items and goods.

    Args:
        query (str): The search query string.

    Returns:
        list: Relevant inventory documents from hybrid search.

    Note:
        Uses both supply_chain_hybrid_search_tool and text_search_index.
    """

    hybrid_search = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store_supply_chain_records,
        search_index_name=TEXT_SEARCH_INDEX,
        top_k=5,
    )

    hybrid_search_result = hybrid_search.get_relevant_documents(query)

    # Remove embedding from the result
    for result in hybrid_search_result:
        if "embedding" in result:
            del result["embedding"]

    return str(hybrid_search_result)

@tool
def update_transit_status(input: str) -> str:
    """

    Updates the status of a shipment in transit.
    Input should be in the format: 'shipment_id,new_status'.
    For example: 'SHP-2024-001,Delayed'

    """
    try:
        shipment_id, new_status = input.split(",")
        # Update the record in MongoDB
        result = collection.update_one(
            {"Contract Number": shipment_id},
            {"$set": {"Current Transit Status": new_status}},
        )
        if result.modified_count > 0:
            return (
                f"Successfully updated status of shipment {shipment_id} to {new_status}"
            )
        return f"No shipment found with ID {shipment_id}"
    except ValueError:
        return "Invalid input format. Please use 'shipment_id,new_status'"
    
@tool
def get_contracts_by_transit_status(status, limit=5):
    """
    Retrieves contract information from MongoDB based on the Current Transit Status.

    Args:
      status: The Current Transit Status to filter by (e.g., "Delayed", "On Schedule").

    Returns:
      A list of contract documents matching the specified status.
    """

    try:
        # Query the collection and project out embedding using $project
        contracts = collection.find(
            {"Current Transit Status": status},
            {
                "Contract Number": 1,
                "Goods Description": 1,
                "Origin": 1,
                "Destination": 1,
                "Shipper": 1,
                "Receiver": 1,
                "Current Transit Status": 1,
            },
        ).limit(limit)

        # Return the results
        return str(list(contracts))
    except Exception as e:
        print(f"An error occurred: {e!s}")
        return []


@tool
def get_contracts_by_inventory_status(status, limit=5):
    """
        Retrieves contract information from MongoDB based on the Inventory Status.

        Args:
            status: The Inventory Status to filter by (e.g., "Delivered", "Awaiting Customs Clearance", "
    In Transit", "Pending Dispatch").
            limit: The maximum number of contracts to retrieve (default is 5).

        Returns:
            A list of contract documents matching the specified inventory status.
    """
    try:
        # Query the collection
        contracts = collection.find(
            {"Inventory Status": status},
            {
                "Contract Number": 1,
                "Goods Description": 1,
                "Origin": 1,
                "Destination": 1,
                "Shipper": 1,
                "Receiver": 1,
                "Inventory Status": 1,
            },
        ).limit(limit)

        # Return the results
        return str(list(contracts))
    except Exception as e:
        print(f"An error occurred: {e!s}")
        return []
    
toolbox = [
    supply_chain_hybrid_search_tool,
    update_transit_status,
    get_contracts_by_transit_status,
    get_contracts_by_inventory_status,
]

#Step 2: Define LLM
# Do note that Anthropic LLM has a rate/token limit
# and this can affect the agentic execution
# https://docs.anthropic.com/en/api/rate-limits#rate-limits
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620", temperature=0, max_tokens=1024, timeout=None
)
#Step 3: Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sender: str

#Step 4: Define Agent Node 
system_message = """
You are an AI-powered Logistics Assistant designed to streamline operations and enhance customer service for an international shipping company. You are equipped with tools to access and process contract details, shipment information, inventory data, and supply chain updates.

Your primary objectives are to:

1. Answer user queries accurately and efficiently by retrieving relevant information from the available data sources.

2. Provide support for operational tasks, including updating shipment statuses and notifying relevant stakeholders.

3. Facilitate informed decision-making by offering insights based on contract terms, shipping regulations, and supply chain dynamics.

Here are some examples of the types of queries and tasks you can handle:

Operational Efficiency:
- "What penalties apply if shipment delays impact our Asia-Europe supply route?"
- "What are the expected delivery timelines for shipments from China to the US?"

Customer Service:
- "What are the customs requirements for electronics shipped to Canada?"
- "What is the status of shipment SHP-2024-003?"

Supply Chain Optimization:
- "What inventory do we have in transit to Europe, and are there any delays?"
- "Which suppliers have the highest on-time delivery rates?"

Shipment Status Update:
- "Update the status of Shipment SHP-2024-001 to 'Delayed' and notify relevant stakeholders."
- "Log a delay for shipment SHP-2024-002 and record the reason for the delay."

Supplier Collaboration:
- "What are the penalties if our supplier delays the shipment?"
- "What are the terms of our contract with supplier XYZ regarding delivery schedules?"

Remember to:

- Always prioritize data accuracy and compliance with regulations.
- Provide clear and concise responses to user queries.
- Utilize the available tools to access and process information effectively.
- Offer helpful insights and recommendations based on your analysis.
- Maintain a professional and customer-centric approach in all interactions.
"""

base_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant, collaborating with other assistants. "
            "Use the provided tools to progress towards answering the question. "
            "If you are unable to fully answer, that's OK, another assistant with different tools "
            "will help where you left off. Execute what you can to make progress. "
            "If you or any of the other assistants have the final answer or deliverable, "
            "prefix your response with FINAL ANSWER so the team knows to stop. "
            "\n{system_message}"
            "\nCurrent time: {time}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt = base_prompt.partial(
    system_message=system_message, time=lambda: str(datetime.now())
)

agent = prompt | llm.bind_tools(toolbox)

name = "SupplyChainAssistant"

def agent_node(state: AgentState, config: RunnableConfig):
    print("----Calling Agent Node-------")
    messages = state["messages"]

    result = agent.invoke(messages, config)

    if isinstance(result, ToolMessage):
        result = ToolMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    elif isinstance(result, AIMessage):
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)

    return {
        "messages": [result],
        "sender": name,
    }

#Step 5: Define Tool Node 
tools_by_name = {tool.name: tool for tool in toolbox}


# TODO: Use AgentState Model for return value
def tool_node(state: AgentState):
    print()
    print("----Calling Tool Node-------")
    print()
    outputs = []
    tool_name = None
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_result = tools_by_name[tool_name].invoke(tool_call["args"])

        print(f"Using tool {tool_name}")
        print(f"Result: {tool_result}")
        print()

        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outputs, "sender": tool_name}

#Step 6: Define Graph
# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    return "continue"
# Create Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chatbot", agent_node)
workflow.add_node("tools", tool_node)

# Set up graph structure
workflow.set_entry_point("chatbot")
workflow.add_conditional_edges(
    "chatbot", should_continue, {"continue": "tools", "end": END}
)
workflow.add_edge("tools", "chatbot")



# Set up MongoDB checkpointer
mongo_client = AsyncIOMotorClient(MONGO_URI)
mongodb_checkpointer = MongoDBSaver(mongo_client, DB_NAME, "state_store")

graph = workflow.compile(checkpointer=mongodb_checkpointer)



#Step8
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

user_input = st.text_area(
    "Enter your query:",
    placeholder="Ask something like 'What are the customs requirements for toys shipped to Canada?'",
    height=150,
)
submit_button = st.button("Submit")

# Function to process user queries through the graph
async def process_query(user_input: str):
    config = {"configurable": {"thread_id": "0"}}
    state = {"messages": [HumanMessage(content=user_input, name="Human")]}

    max_retries = 3
    retry_delay = 1

    response = []

    for attempt in range(max_retries):
        try:
            async for chunk in graph.astream(state, config, stream_mode="values"):
                if chunk.get("messages"):
                    last_message = chunk["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        last_message.name = last_message.name or "AI"
                        response.append(last_message.content)
                    elif isinstance(last_message, ToolMessage):
                        response.append(
                            str(
                    f"Content: {last_message.content}\n"
                )
                        )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"An unexpected error occurred: {e!s}. Retrying...")
                retry_delay *= 2
                await asyncio.sleep(retry_delay)
            else:
                st.error(f"Max retries reached. API error: {e!s}")
                break

    return "\n".join(map(str, response))


# Handle user submission
if submit_button and user_input.strip():
    # Inform the user that the assistant is processing the query
    with st.spinner("Processing your query, please wait..."):
        # Run the async function to process the query
        response = asyncio.run(process_query(user_input))
        if response:
            st.success("Assistant Response:")
            st.write(response)
else:
    if submit_button:
        st.warning("Please enter a valid query.", icon="⚠")