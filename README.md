# 🚢 AI Agentic Supply Chain System

## 📄 Project Overview
The **AI Agentic Supply Chain System** is designed to **streamline international shipping operations and contract management** by leveraging **AI-driven agents, Retrieval-Augmented Generation (RAG) architecture, MongoDB Atlas, and Large Language Models (LLMs)**. This intelligent system enhances contract management, optimizes supply chain workflows, and improves customer service for global shipping companies.

This project is based on the original implementation from [MongoDB Developer GenAI Showcase](https://github.com/mongodb-developer/GenAI-Showcase/blob/main/partners/gravity9/Agentic_System_Enhanced_Contract_and_Supply_Chain_Management_for_International_Shipping.ipynb), where additional features such as **Streamlit-based UI** have been incorporated for an enhanced user experience.

## 🎯 Objective
- Automate supply chain and contract management using AI-driven assistants.
- Provide **real-time shipment updates** and contract insights.
- Improve efficiency, reduce operational delays, and enhance **customer service**.

---

## 🛠️ Key Features
- 🔎 **Retrieve Contract & Shipment Data** – Search MongoDB-stored contracts, inventory, and routing details.
- 🧠 **AI-Powered Queries** – Use **LLMs + RAG architecture** for natural language interactions.
- 🌐 **Streamlit UI** – Interactive web interface for users to interact with the system easily.
- 📊 **Real-Time Shipment Updates** – Modify shipment statuses and notify stakeholders instantly.
- 🚀 **Hybrid Search Capabilities** – Perform **vector + full-text searches** for contract and inventory data.
- 📈 **Actionable Insights** – Generate insights for supply chain optimization and compliance.

---

## 🏗️ Solution Overview
International shipping contracts contain detailed clauses covering **tariffs, insurance, delivery timelines, and penalties**. This AI-powered system enables **operations, legal, and supply chain teams** to access accurate contract details quickly. 

### 🔹 **AI Agent Capabilities**
1. **Retrieve Relevant Data** – Searches **MongoDB Atlas** for contract terms, shipment records, inventory levels, and supplier agreements.
2. **Generate Natural Language Responses** – Converts complex legal terms and logistics data into simple answers for non-technical users.
3. **Update Shipment Statuses** – Modifies shipment details, logs delays, and notifies impacted teams.
4. **Streamlit Web UI** – Allows users to interact with the AI assistant through a simple web interface.

---

## 🏢 Use Case Examples
- 📦 **Operations Management**: "What penalties apply for shipment delays on the Asia-Europe route?"
- 🏛️ **Legal & Compliance**: "What are the customs requirements for shipping electronics to Canada?"
- 🚛 **Supply Chain Optimization**: "What inventory is in transit to Europe, and are there delays?"
- 🤝 **Supplier Collaboration**: "What are the penalties if our supplier delays shipment?"
- 🚢 **Shipment Tracking**: "Update Shipment **SHP-2024-001** to 'Delayed' and notify relevant teams."

---
### 🏢 AI Agent Interactive Example
**User:** Hi there  
**Assistant:** ----Calling Agent Node-------  
Hello! Welcome to our AI-powered Logistics Assistant. I'm here to help you with any questions or tasks related to our international shipping operations, contract details, shipment information, inventory data, and supply chain updates. How may I assist you today?  

Is there a specific area you'd like information on, such as:
1. Checking the status of a shipment
2. Inquiring about contract details
3. Getting updates on inventory
4. Requesting information about shipping routes or customs requirements
5. Updating shipment statuses
6. Any other logistics-related query

**User:** What can you do?  
**Assistant:** ----Calling Agent Node-------  
Certainly! I'd be happy to explain my capabilities as an AI-powered Logistics Assistant. I'm designed to help with various aspects of international shipping and logistics. Here's an overview of what I can do:

1. **Information Retrieval:**
   - Search and provide details on inventory, contracts, and shipments.
   - Retrieve contract information based on transit or inventory status.

2. **Shipment Status Management:**
   - Update and track shipment statuses.

3. **Contract and Inventory Analysis:**
   - Analyze contract terms and inventory levels.

4. **Supply Chain Optimization:**
   - Provide details on shipping routes, expected delivery timelines, and potential delays.

5. **Customer Service Support:**
   - Answer customs-related queries and provide shipment ETAs.

6. **Operational Efficiency:**
   - Provide insights on penalties for delays and contract breaches.

7. **Regulatory Compliance:**
   - Offer guidance on shipping regulations and requirements.

To use these capabilities, simply ask me questions like:
- "What's the status of shipment SHP-2024-001?"
- "Are there any delayed shipments to Europe?"
- "What are the customs requirements for electronics shipped to Canada?"
- "Update the status of Shipment SHP-2024-002 to 'Delayed'"
- "What are the terms in our contract with supplier XYZ regarding delivery schedules?"

Let me know how I can assist you further!

---
## 🛡️ Security Considerations
- **Role-Based Access Control** – Limit who can update shipment statuses.
- **Data Encryption** – Ensure contract data is stored securely.
- **Audit Logging** – Track all agent interactions for compliance.

## 📈 Monitoring & Logging
- **MongoDB Aggregation Pipelines** – Monitor query trends and usage.
- **Logging Mechanisms** – Keep records of queries and updates for audit trails.

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

**⭐ Don't forget to star the repository if you found this useful!** 🚀
