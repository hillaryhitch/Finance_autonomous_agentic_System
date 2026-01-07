import os
import boto3
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.models.aws import AwsBedrock
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.exa import ExaTools
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.skills import Skills, LocalSkills

# Initialize Boto3 session
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name="us-east-1"
)

# Clean base model (NO skills parameter for AwsBedrock)
model_base = AwsBedrock(
    id="anthropic.claude-3-sonnet-20240229-v1:0",#"anthropic.claude-3-5-sonnet-20241022-v2:0",
    session=session
)

# Knowledge base setup
knowledge = Knowledge(
    vector_db=ChromaDb(
        name="safaricom_finance_team",
        path="./chroma_db",
        embedder=HuggingfaceCustomEmbedder(
            id="sentence-transformers/all-mpnet-base-v2"
        ),
    ),
    readers=[PDFReader(path="finance_data/safaricom_docs", chunk=True)],
)

# Shared Safaricom skills (loaded per-agent as needed)
def safaricom_skills(role_specific_folders=None):
    """Helper to create role-specific skills"""
    base_folders = [
        "./skills/financial-metrics",
        "./skills/mpesa-financial-services", 
        "./skills/safaricom-telco-expertise",
    ]
    if role_specific_folders:
        base_folders.extend(role_specific_folders)
    return Skills(loaders=[LocalSkills(folder) for folder in base_folders])

# AI & Innovation Officer
ai_innovation_officer = Agent(
    name="AI & Innovation Officer",
    role="AI Strategy & Innovation",
    model=model_base,
    skills=safaricom_skills(["./skills/safaricom-ai-innovation"]),
    tools=[ExaTools(), DuckDuckGoTools()],
    knowledge=knowledge,
    instructions=[
        "You are the AI & Innovation Officer at Safaricom, responsible for driving AI strategy and innovation across finance and operations.",
        "Key Responsibilities: Identify AI/ML opportunities, develop fraud detection solutions, lead M-PESA innovation, ensure ethical AI.",
        "Document Creation: Use get_skill_instructions('pptx') for presentations, get_skill_instructions('xlsx') for ROI models, get_skill_instructions('docx') for strategy papers.",
        "Focus on M-PESA fraud detection, Ethiopia market intelligence, and finance automation.",
    ],
    add_datetime_to_context=True,
    markdown=True,
)

# Financial Reporting Lead
financial_reporting_agent = Agent(
    name="Financial Reporting Lead",
    role="Financial Reporting & Compliance",
    model=model_base,
    skills=safaricom_skills(["./skills/financial-reporting"]),  # Add this folder if needed
    knowledge=knowledge,
    tools=[ExaTools()],
    instructions=[
        "You are the Financial Reporting Lead at Safaricom, responsible for accurate financial reporting per IFRS.",
        "Handle quarterly/annual statements, audit coordination, NSE compliance, M-PESA revenue recognition.",
        "Document Creation: Use get_skill_instructions('xlsx') for financial statements, get_skill_instructions('pptx') for board presentations.",
        "Segments: Mobile, M-PESA, Fixed, Enterprise. Track EBITDA, capex, Ethiopia consolidation.",
    ],
    add_datetime_to_context=True,
    markdown=True,
)

# Business Case Analyst
business_case_agent = Agent(
    name="Business Case Analyst",
    role="Business Case Development & Evaluation",
    model=model_base,
    skills=safaricom_skills(["./skills/business-case"]),
    knowledge=knowledge,
    tools=[DuckDuckGoTools(), ExaTools()],
    instructions=[
        "You are the Business Case Analyst at Safaricom, evaluating investments and strategic initiatives.",
        "Develop NPV/IRR models, assess M-PESA expansion, Ethiopia 5G rollout, enterprise growth.",
        "Document Creation: Use get_skill_instructions('xlsx') for financial models with sensitivity analysis, get_skill_instructions('pptx') for executive presentations.",
    ],
    add_datetime_to_context=True,
    markdown=True,
)

# Budget Planning Manager
budget_planning_agent = Agent(
    name="Budget Planning Manager",
    role="Budgeting & Financial Planning",
    model=model_base,
    skills=safaricom_skills(["./skills/budget-planning"]),
    knowledge=knowledge,
    tools=[],
    instructions=[
        "You are the Budget Planning Manager at Safaricom, leading annual budgeting and forecasting.",
        "Handle revenue/opex/capex allocation across Kenya/Ethiopia, variance analysis.",
        "Document Creation: Use get_skill_instructions('xlsx') for budget templates with pivots, get_skill_instructions('pptx') for reviews.",
    ],
    add_datetime_to_context=True,
    markdown=True,
)

# Treasury Manager
treasury_agent = Agent(
    name="Treasury Manager",
    role="Treasury & Cash Management",
    model=model_base,
    skills=safaricom_skills(["./skills/treasury"]),
    knowledge=knowledge,
    tools=[DuckDuckGoTools()],
    instructions=[
        "You are the Treasury Manager at Safaricom, managing liquidity, M-PESA float, currency risks.",
        "Forecast cash flows, manage KES/USD/ETB exposure, dividend payments.",
        "Document Creation: Use get_skill_instructions('xlsx') for cash flow forecasts, get_skill_instructions('pptx') for treasury reports.",
    ],
    add_datetime_to_context=True,
    markdown=True,
)

# Senior Financial Analyst
financial_analyst_agent = Agent(
    name="Senior Financial Analyst",
    role="Financial Analysis & Strategic Insights",
    model=model_base,
    skills=safaricom_skills(["./skills/financial-analysis"]),
    knowledge=knowledge,
    tools=[DuckDuckGoTools(), ExaTools()],
    instructions=[
        "You are the Senior Financial Analyst at Safaricom, providing data-driven strategic insights.",
        "Analyze KPIs (ARPU, churn, EBITDA), competitive benchmarking, Ethiopia performance.",
        "Document Creation: Use get_skill_instructions('xlsx') for KPI dashboards, get_skill_instructions('pptx') for strategy decks.",
    ],
    add_datetime_to_context=True,
    markdown=True,
)

# CFO Team
safaricom_finance_team = Team(
    name="Chief Financial Officer",
    model=model_base,
    instructions=[
        "You are the CFO of Safaricom PLC. Delegate tasks to specialized offices:",
        "- Reporting/Compliance → Financial Reporting Lead",
        "- Business cases/Investments → Business Case Analyst",
        "- Budgeting/Forecasting → Budget Planning Manager", 
        "- Cash/Liquidity → Treasury Manager",
        "- Analysis/Insights → Senior Financial Analyst",
        "- AI/Innovation → AI & Innovation Officer",
        "Maintain financial discipline and strategic focus on Ethiopia, M-PESA, 5G.",
    ],
    members=[
        financial_reporting_agent,
        business_case_agent,
        budget_planning_agent,
        treasury_agent,
        financial_analyst_agent,
        ai_innovation_officer,
    ],
    add_datetime_to_context=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)

# File download helper (adapt as needed)
def download_files_from_response(response):
    """Placeholder - replace with your file handling logic"""
    print("✅ Response received. Files would be processed here.")
    print(f"Response summary: {response.content[:200]}...")
    return []

if __name__ == "__main__":
    # Test the full team
    print("\n" + "="*80)
    print("Q4 2025 Financial Report Package")
    print("="*80)
    response1 = safaricom_finance_team.run(
        """Create Q4 2025 financial report package:
        1. EXCEL: Revenue Mobile KES 180B, M-PESA KES 95B, EBITDA KES 150B (50% margin)
        2. POWERPOINT: Executive summary, financial highlights, segment comparison
        3. WORD: IFRS 15 notes for M-PESA"""
    )
    download_files_from_response(response1)
    
    print("\n" + "="*80)
    print("Ethiopia 5G Business Case")
    print("="*80)
    response2 = safaricom_finance_team.run(
        """Ethiopia 5G business case:
        1. EXCEL: 5yr projections, Year1 revenue USD 50M (40% growth), Capex USD 200M, NPV/IRR
        2. POWERPOINT: Market opportunity, financials, risks, recommendation"""
    )
    download_files_from_response(response2)


# from agno.agent import Agent
# from agno.knowledge.knowledge import Knowledge
# from agno.knowledge.reader.pdf_reader import PDFReader
# from agno.models.anthropic import Claude
# from agno.team import Team
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.tools.exa import ExaTools
# from agno.vectordb.chroma import ChromaDb
# from anthropic import Anthropic
# from anthropic.lib import files_from_dir
# from agno.knowledge.embedder.huggingface import HuggingfaceCustomEmbedder

# import os
# from anthropic.lib import files_from_dir

# # Initialize Anthropic client
# # anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# # # Register custom skills with Anthropic
# # print("Registering custom skills with Anthropic...")

# # safaricom_telco_skill = anthropic_client.beta.skills.create(
# #     display_title="Safaricom Telecommunications Expertise.",
# #     files=files_from_dir("skills/safaricom-telco-expertise"),
# #     betas=["skills-2025-10-02"]
# # )

# # mpesa_skill = anthropic_client.beta.skills.create(
# #     display_title="M-PESA Financial Services.",
# #     files=files_from_dir("skills/mpesa-financial-services"),
# #     betas=["skills-2025-10-02"]
# # )

# # financial_metrics_skill = anthropic_client.beta.skills.create(
# #     display_title="Safaricom Financial Metrics.",
# #     files=files_from_dir("skills/financial-metrics"),
# #     betas=["skills-2025-10-02"]
# # )

# # ai_skill = anthropic_client.beta.skills.create(
# #     display_title="AI skills",
# #     files=files_from_dir("skills/safaricom-ai-innovation"),
# #     betas=["skills-2025-10-02"]
# # )


# import boto3
# import anthropic
# import os

# # Initialize the Bedrock runtime client
# # The Anthropic SDK automatically uses boto3 under the hood when configured correctly
# # Ensure the region matches where you have model access
# import boto3
# from agno.agent import Agent
# from agno.models.aws import AwsBedrock
# from agno.skills import Skills, LocalSkills

# # 1. Initialize the Boto3 Session with temporary credentials
# session = boto3.Session(
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
#     region_name="us-east-1"
# )


# # Initialize the Anthropic client with the Bedrock client
# anthropic_client = anthropic.AnthropicBedrock(aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN"))

# # Initialize Anthropic client
# # anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# # def get_or_create_skill(client, display_title, files_path):
# #     """Get existing skill or create new one if it doesn't exist"""
# #     try:
# #         # List existing skills
# #         existing_skills = client.beta.skills.list(betas=["skills-2025-10-02"])
        
# #         # Check if skill with this title already exists
# #         for skill in existing_skills.data:
# #             if skill.display_title == display_title:
# #                 print(f"✓ Found existing skill: {display_title} (ID: {skill.id})")
# #                 return skill
        
# #         # If not found, create new skill
# #         print(f"Creating new skill: {display_title}...")
# #         skill = client.beta.skills.create(
# #             display_title=display_title,
# #             files=files_from_dir(files_path),
# #             betas=["skills-2025-10-02"]
# #         )
# #         print(f"✓ Created skill: {display_title} (ID: {skill.id})")
# #         return skill
        
# #     except Exception as e:
# #         print(f"Error with skill '{display_title}': {e}")
# #         raise

# # # Register custom skills
# # print("\n" + "="*60)
# # print("REGISTERING CUSTOM SKILLS")
# # print("="*60)

# # safaricom_telco_skill = get_or_create_skill(
# #     anthropic_client,
# #     "Safaricom Telecommunications Expertise.",
# #     "skills/safaricom-telco-expertise"
# # )

# # mpesa_skill = get_or_create_skill(
# #     anthropic_client,
# #     "M-PESA Financial Services.",
# #     "skills/mpesa-financial-services"
# # )

# # financial_metrics_skill = get_or_create_skill(
# #     anthropic_client,
# #     "Safaricom Financial Metrics.",
# #     "skills/financial-metrics"
# # )

# # ai_skill = get_or_create_skill(
# #     anthropic_client,
# #     "AI skills",
# #     "skills/safaricom-ai-innovation"
# # )





# # Model with document generation skills
# # model_with_docs = Claude(
# #     id="claude-sonnet-4-20250514",
# #     skills=[
# #         {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
# #         {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
# #         {"type": "anthropic", "skill_id": "docx", "version": "latest"},
# #     ]
# # )

# model_with_docs = AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session,
#     skills=[
#         {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "docx", "version": "latest"},
#     ])

# # Model without skills for analysis-heavy tasks
# model_base = AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session)#Claude(id="claude-sonnet-4-20250514")


# # 1. Update your Knowledge setup with proper embeddings
# knowledge = Knowledge(
#     vector_db=ChromaDb(
#         name="safaricom_finance_team",
#         path="./chroma_db",
#         embedder=HuggingfaceCustomEmbedder(
#             id="sentence-transformers/all-mpnet-base-v2"
#         ),
#     ),
#     readers=[PDFReader(path="finance_data/safaricom_docs", chunk=True)],
# )




# # # ChromaDB configuration for Safaricom Finance knowledge base
# # knowledge = Knowledge(
# #     vector_db=ChromaDb(
# #         name="safaricom_finance_team",
# #         path="./chroma_db",
# #     ),
# # )

# # # Add Safaricom-specific financial documents, policies, and reports
# # knowledge.add_content(
# #     path="finance_data/safaricom_docs", reader=PDFReader(chunk=True)
# # )

# # Safaricom-specific context to be added to agent instructions
# # NOTE: Custom skills in Anthropic require a proper skill folder structure
# # For now, we'll add this context directly to agent instructions instead




# # AI & Innovation Office
# ai_innovation_officer = Agent(
#     name="AI & Innovation Officer",
#     role="AI Strategy & Innovation",
#     # model=Claude(
#     # id="claude-sonnet-4-20250514",
#     model=AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session,
# ),
# skills=Skills(loaders=[
#     LocalSkills("./skills/safaricom-ai-innovation")
# ]),
#     tools=[ExaTools(), DuckDuckGoTools()],
#     knowledge=knowledge,
#     instructions=[
#         "You are the AI & Innovation Officer at Safaricom, responsible for driving AI strategy and innovation across finance and operations.",
#         "Key Responsibilities:",
#         "1. Identify AI/ML opportunities to improve financial processes and decision-making",
#         "2. Develop AI-powered solutions for fraud detection, credit risk assessment, and revenue optimization",
#         "3. Lead innovation initiatives for M-PESA and enterprise services",
#         "4. Collaborate with technology and business teams on AI implementation",
#         "5. Monitor AI trends in telecom and fintech sectors",
#         "6. Assess ROI and business impact of AI investments",
#         "7. Ensure ethical AI practices and data governance",
#         "8. Drive automation of financial reporting and analysis",
#         "",
#         "Document Creation:",
#         "- Create PowerPoint presentations for innovation proposals and AI roadmaps",
#         "- Generate Excel models for ROI analysis and AI project tracking",
#         "- Produce Word documents for AI strategy papers and guidelines",
#         "",
#         "Safaricom Context:",
#         "- Focus on AI solutions that enhance M-PESA fraud detection and credit scoring",
#         "- Leverage network data for predictive analytics and customer insights",
#         "- Support Ethiopia market expansion with AI-driven market intelligence",
#         "- Collaborate with finance offices on automation and efficiency gains",
#     ],
#     add_datetime_to_context=True,
#     markdown=True,
# )

# # Financial Reporting Office
# financial_reporting_agent = Agent(
#     name="Financial Reporting Lead",
#     role="Financial Reporting & Compliance",
#     # model=Claude(
#     # id="claude-sonnet-4-20250514",
#     model=AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session,
#     skills=[
#         {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
#         #  {"type": "custom", "skill_id": financial_metrics_skill.id, "version": "latest"},
        
#     ]),
#     skills=Skills(loaders=[
#         LocalSkills("./skills/financial-metrics"),
#     ]),
#     knowledge=knowledge,
#     tools=[ExaTools()],
#     instructions=[
#         "You are the Financial Reporting Lead at Safaricom, responsible for accurate and timely financial reporting.",
#         "Key Responsibilities:",
#         "1. Prepare quarterly and annual financial statements per IFRS standards",
#         "2. Ensure compliance with NSE listing requirements and reporting deadlines",
#         "3. Coordinate with external auditors and manage audit processes",
#         "4. Oversee revenue recognition for telecom services and M-PESA transactions",
#         "5. Report on segment performance (Mobile, M-PESA, Fixed, Enterprise)",
#         "6. Maintain internal controls and financial reporting accuracy",
#         "7. Prepare management reports and board presentations",
#         "8. Monitor regulatory reporting requirements (CA, CBK, KRA)",
#         "",
#         "Document Creation:",
#         "- Generate Excel financial statements and segment reporting templates",
#         "- Create PowerPoint presentations for board meetings and investor relations",
#         "- Produce Word documents for audit reports and compliance summaries",
#         "- Always include proper formatting, headers, and professional layouts",
#         "",
#         "Safaricom Context:",
#         "- Segment reporting: Mobile Services, M-PESA, Fixed Services, Enterprise",
#         "- Key line items: Service revenue, capex, EBITDA, free cash flow",
#         "- Ethiopia operations: Consolidate STE financials, currency translation",
#         "- M-PESA financials: Transaction revenue, loan book quality, credit losses",
#     ],
#     add_datetime_to_context=True,
#     markdown=True,
# )

# # Business Case Office
# business_case_agent = Agent(
#     name="Business Case Analyst",
#     role="Business Case Development & Evaluation",
#     # model=Claude(
#     # id="claude-sonnet-4-20250514",
#     model=AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session,
#     skills=[
#         {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
#         # {"type": "custom", "skill_id":financial_metrics_skill.id, "version": "latest"},
        
#     ]),

#     skills=Skills(loaders=[
#         LocalSkills("./skills/financial-metrics")
#     ]),
#     knowledge=knowledge,
#     tools=[DuckDuckGoTools(), ExaTools()],
#     instructions=[
#         "You are the Business Case Analyst at Safaricom, responsible for evaluating investment opportunities and strategic initiatives.",
#         "Key Responsibilities:",
#         "1. Develop comprehensive business cases for capital investments and strategic projects",
#         "2. Conduct financial modeling (NPV, IRR, sensitivity analysis) for investment decisions",
#         "3. Assess market opportunities and competitive positioning",
#         "4. Evaluate risks and develop mitigation strategies",
#         "5. Analyze M-PESA product expansion opportunities (lending, savings, insurance)",
#         "6. Support Ethiopia market entry and expansion decisions",
#         "7. Collaborate with strategy, product, and technology teams",
#         "8. Present investment recommendations to CFO and executive leadership",
#         "",
#         "Document Creation:",
#         "- Create Excel financial models with NPV/IRR calculations, sensitivity tables, and charts",
#         "- Generate PowerPoint business case presentations with executive summaries",
#         "- Produce Word documents for detailed business case write-ups and risk assessments",
#         "- Include assumptions, scenarios (base/optimistic/pessimistic), and recommendations",
#         "",
#         "Safaricom Context:",
#         "- Major capital decisions: 5G network rollout, fiber expansion, Ethiopia investment",
#         "- M-PESA innovations: New financial products, regional expansion, merchant solutions",
#         "- Enterprise growth: IoT solutions, cloud services, cybersecurity offerings",
#         "- Consider regulatory requirements and spectrum costs in business cases",
#     ],
#     add_datetime_to_context=True,
#     markdown=True,
# )

# # Budget Office
# budget_planning_agent = Agent(
#     name="Budget Planning Manager",
#     role="Budgeting & Financial Planning",
#     # model=Claude(
#     # id="claude-sonnet-4-20250514",

#     model=AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session,
#     skills=[
#         {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "docx", "version": "latest"}
#     ]),
#     skills=Skills(loaders=[
#         LocalSkills("./skills/financial-metrics"),
#         LocalSkills("./skills/mpesa-financial-services"),
#         LocalSkills("./skills/safaricom-telco-expertise"),
#     ]),
#     knowledge=knowledge,
#     tools=[],
#     instructions=[
#         "You are the Budget Planning Manager at Safaricom, responsible for budgeting, forecasting, and financial planning.",
#         "Key Responsibilities:",
#         "1. Lead the annual budgeting process across all business units",
#         "2. Develop revenue and expense forecasts based on market trends and business drivers",
#         "3. Allocate resources to support strategic priorities",
#         "4. Monitor budget performance and conduct variance analysis",
#         "5. Prepare rolling forecasts and update financial projections",
#         "6. Collaborate with business units on budget submissions and reviews",
#         "7. Support capex planning for network and IT infrastructure",
#         "8. Provide financial planning insights to CFO and leadership team",
#         "",
#         "Document Creation:",
#         "- Create Excel budget templates with formulas, pivot tables, and budget vs. actual tracking",
#         "- Generate PowerPoint presentations for budget reviews and variance explanations",
#         "- Produce Word documents for budget guidelines and planning assumptions",
#         "- Include monthly/quarterly phasing and departmental breakdowns",
#         "",
#         "Safaricom Context:",
#         "- Revenue streams: Mobile voice/data/SMS, M-PESA, fixed broadband, enterprise",
#         "- Major cost items: Network operations, staff costs, marketing, license fees",
#         "- Capex planning: 5G rollout, fiber expansion, Ethiopia infrastructure",
#         "- Ethiopia budgeting: Separate P&L, currency considerations, investment phase",
#         "- Consider regulatory costs: Spectrum fees, CA levies, tax obligations",
#     ],
#     add_datetime_to_context=True,
#     markdown=True,
# )

# # Treasury & Cash Management Office
# treasury_agent = Agent(
#     name="Treasury Manager",
#     role="Treasury & Cash Management",
#     # model=Claude(
#     # id="claude-sonnet-4-20250514",
    
#     model=AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session,
    
#     skills=[
#         {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "xlsx", "version": "latest"}
#     ]),
 
#     skills=Skills(loaders=[
#         LocalSkills("./skills/financial-metrics"),
#         LocalSkills("./skills/mpesa-financial-services"),
#         LocalSkills("./skills/safaricom-telco-expertise"),
#     ]),
#     knowledge=knowledge,
#     tools=[DuckDuckGoTools()],
#     instructions=[
#         "You are the Treasury Manager at Safaricom, responsible for cash management, liquidity, and financial risk management.",
#         "Key Responsibilities:",
#         "1. Manage daily liquidity and cash positions across Kenya and Ethiopia",
#         "2. Oversee M-PESA float management and ensure adequate liquidity for mobile money operations",
#         "3. Monitor and manage currency exposure (KES, USD, ETB)",
#         "4. Optimize working capital and short-term investments",
#         "5. Manage debt facilities, refinancing, and credit rating relationships",
#         "6. Conduct cash flow forecasting and liquidity planning",
#         "7. Implement hedging strategies for currency and interest rate risks",
#         "8. Maintain banking relationships and negotiate facilities",
#         "",
#         "Document Creation:",
#         "- Create Excel cash flow forecasts with daily/weekly/monthly projections",
#         "- Generate PowerPoint presentations for treasury committee meetings",
#         "- Produce Word documents for treasury policies and hedging strategies",
#         "- Include currency exposure analysis and risk mitigation plans",
#         "",
#         "Safaricom Context:",
#         "- M-PESA liquidity: Critical to ensure seamless customer transactions",
#         "- Multi-currency operations: KES (Kenya), ETB (Ethiopia), USD (international settlements)",
#         "- Ethiopia investment: Large USD/ETB cash requirements for network build-out",
#         "- Dividend payments: Significant cash outflows, typically paid bi-annually",
#         "- Debt profile: Bank loans, potential bond issuances, maintain investment-grade rating",
#     ],
#     add_datetime_to_context=True,
#     markdown=True,
# )

# # Financial Analysis & Strategy Office
# financial_analyst_agent = Agent(
#     name="Senior Financial Analyst",
#     role="Financial Analysis & Strategic Insights",
#     # model=Claude(
#     # id="claude-sonnet-4-20250514",
#     model=AwsBedrock(
#     id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     session=session,
#     skills=[
#         {"type": "anthropic", "skill_id": "pptx", "version": "latest"},
#         {"type": "anthropic", "skill_id": "xlsx", "version": "latest"}
#     ]),
#     skills=Skills(loaders=[
#         LocalSkills("./skills/financial-metrics"),
#         LocalSkills("./skills/mpesa-financial-services"),
#         LocalSkills("./skills/safaricom-telco-expertise"),
#     ]),
#     knowledge=knowledge,
#     tools=[DuckDuckGoTools(), ExaTools()],
#     instructions=[
#         "You are the Senior Financial Analyst at Safaricom, providing strategic financial insights and analysis.",
#         "Key Responsibilities:",
#         "1. Conduct in-depth financial analysis of business performance and trends",
#         "2. Monitor and analyze key financial and operational metrics (ARPU, churn, EBITDA)",
#         "3. Perform competitive benchmarking and market analysis",
#         "4. Support strategic decision-making with data-driven insights",
#         "5. Analyze M-PESA performance and growth opportunities",
#         "6. Evaluate Ethiopia market performance and investment returns",
#         "7. Prepare executive dashboards and strategic presentations",
#         "8. Identify risks and opportunities across the business",
#         "",
#         "Document Creation:",
#         "- Create Excel dashboards with KPIs, trends, and competitive benchmarks",
#         "- Generate PowerPoint strategy presentations with charts and insights",
#         "- Produce Word documents for detailed analysis reports and market assessments",
#         "- Include visual data storytelling with charts, tables, and commentary",
#         "",
#         "Safaricom Context:",
#         "- Track segment performance: Mobile, M-PESA, Fixed, Enterprise, Ethiopia",
#         "- Monitor competitive dynamics: Airtel pricing, Telkom strategy, Ethio Telecom moves",
#         "- Ethiopia focus: Track subscriber growth, revenue ramp, capex deployment, breakeven path",
#         "- M-PESA analytics: Transaction trends, product adoption, merchant growth, credit quality",
#         "- Strategic themes: Digital transformation, platform economics, regional expansion",
#     ],
#     add_datetime_to_context=True,
#     markdown=True,
# )

# # CFO Office - Team Leader
# safaricom_finance_team = Team(
#     name="Chief Financial Officer",
#     model=model_base,
#     instructions=[
#         "You are the Chief Financial Officer (CFO) of Safaricom PLC, responsible for overall financial leadership and strategic decision-making.",
#         "Always delegate tasks to the appropriate office based on expertise and context.",
#         "",
#         "Key Responsibilities:",
#         "1. Provide strategic financial leadership and oversee all finance functions",
#         "2. Drive financial performance and shareholder value creation",
#         "3. Oversee financial planning, reporting, and analysis",
#         "4. Manage capital allocation and investment decisions",
#         "5. Ensure financial compliance and governance",
#         "6. Support CEO and Board on strategic and financial matters",
#         "7. Lead investor relations and stakeholder communications",
#         "8. Champion innovation and AI/digital transformation in finance",
#         "",
#         "Delegation Guidelines:",
#         "1. Financial Reporting & Compliance:",
#         "   - Quarterly/annual reports → Financial Reporting Lead",
#         "   - Audit coordination → Financial Reporting Lead",
#         "   - IFRS/regulatory questions → Financial Reporting Lead",
#         "",
#         "2. Strategic Planning & Investments:",
#         "   - Business cases and investment analysis → Business Case Analyst",
#         "   - Capital allocation decisions → Business Case Analyst + Financial Analyst",
#         "   - M&A opportunities → Business Case Analyst + Financial Analyst",
#         "",
#         "3. Budgeting & Planning:",
#         "   - Annual budget process → Budget Planning Manager",
#         "   - Forecasting and variance analysis → Budget Planning Manager",
#         "   - Resource allocation → Budget Planning Manager",
#         "",
#         "4. Treasury & Risk:",
#         "   - Cash management and liquidity → Treasury Manager",
#         "   - Currency/debt management → Treasury Manager",
#         "   - M-PESA float management → Treasury Manager",
#         "",
#         "5. Analysis & Insights:",
#         "   - Performance analysis → Senior Financial Analyst",
#         "   - Competitive intelligence → Senior Financial Analyst",
#         "   - Strategic insights → Senior Financial Analyst",
#         "",
#         "6. Innovation & AI:",
#         "   - AI strategy and innovation → AI & Innovation Officer",
#         "   - Process automation → AI & Innovation Officer",
#         "   - Digital transformation → AI & Innovation Officer",
#         "",
#         "Safaricom Strategic Priorities:",
#         "- Ethiopia market: Scale operations, path to profitability, manage investment",
#         "- M-PESA growth: Expand product suite, deepen penetration, regional expansion",
#         "- Network leadership: 5G rollout, fiber expansion, quality leadership",
#         "- Enterprise growth: IoT, cloud, digital solutions for businesses",
#         "- Shareholder returns: Maintain strong dividends, deliver consistent growth",
#         "- Digital transformation: Leverage AI/ML, automate processes, enhance analytics",
#         "",
#         "Always maintain financial discipline, strategic focus, and stakeholder value creation.",
#     ],
#     members=[
#         financial_reporting_agent,
#         business_case_agent,
#         budget_planning_agent,
#         treasury_agent,
#         financial_analyst_agent,
#         ai_innovation_officer,
#     ],
#     add_datetime_to_context=True,
#     markdown=True,
#     debug_mode=True,
#     show_members_responses=True,
# )


# # Helper function to download generated files
# def download_files_from_response(response):
#     """Download files from agent response"""
#     from file_download_helper import download_skill_files
    
#     if response.messages:
#         for msg in response.messages:
#             if hasattr(msg, "provider_data") and msg.provider_data:
#                 files = download_skill_files(msg.provider_data, anthropic_client)
#                 if files:
#                     print(f"\n✅ Downloaded {len(files)} file(s):")
#                     for file in files:
#                         print(f"  📄 {file}")
#                     return files
#     return []


# if __name__ == "__main__":
#     # Example 1: Financial reporting with document generation
#     print("\n" + "="*80)
#     print("EXAMPLE 1: Q4 Financial Report Package")
#     print("="*80)
#     response1 = safaricom_finance_team.run(
#         """Create a comprehensive Q4 2025 financial report package:
        
#         1. EXCEL FILE (Q4_2025_Financials.xlsx):
#            - Revenue by segment: Mobile KES 180B, M-PESA KES 95B, Enterprise KES 25B
#            - EBITDA: KES 150B (50% margin)
#            - Include formulas and a revenue breakdown chart
        
#         2. POWERPOINT (Q4_2025_Board_Presentation.pptx):
#            - Executive summary slide
#            - Financial highlights slide with key metrics
#            - Segment performance comparison
           
#         3. WORD DOCUMENT (Q4_2025_IFRS_Notes.docx):
#            - Brief IFRS 15 revenue recognition notes for M-PESA transactions
#            - Summary of key accounting policies applied"""
#     )
#     download_files_from_response(response1)

#     # Example 2: Business case with financial model
#     print("\n" + "="*80)
#     print("EXAMPLE 2: Ethiopia 5G Business Case")
#     print("="*80)
#     response2 = safaricom_finance_team.run(
#         """Develop a business case for Ethiopia 5G network rollout:
        
#         1. EXCEL MODEL (Ethiopia_5G_Business_Case.xlsx):
#            - 5-year projections: Year 1 revenue USD 50M, growing 40% annually
#            - Capex: USD 200M upfront, USD 30M annually
#            - Calculate NPV, IRR, and payback period
#            - Include sensitivity analysis table
           
#         2. POWERPOINT (Ethiopia_5G_Presentation.pptx):
#            - Market opportunity slide
#            - Financial projections summary
#            - Risk assessment and mitigation
#            - Investment recommendation"""
#     )
#     download_files_from_response(response2)

#     # Example 3: Budget package
#     print("\n" + "="*80)
#     print("EXAMPLE 3: FY2026 Budget Package")
#     print("="*80)
#     response3 = safaricom_finance_team.run(
#         """Create FY2026 budget documentation:
        
#         1. EXCEL (FY2026_Budget.xlsx):
#            - Revenue budget by quarter and segment
#            - Opex budget: Network KES 80B, Staff KES 25B, Marketing KES 15B
#            - Capex allocation: Kenya 5G KES 40B, Fiber KES 20B, Ethiopia KES 30B
#            - Include variance analysis template
           
#         2. WORD (FY2026_Budget_Assumptions.docx):
#            - Key planning assumptions
#            - ARPU and subscriber growth forecasts
#            - Risk factors and contingency plans"""
#     )
#     download_files_from_response(response3)

#     # Example 4: AI innovation proposal
#     print("\n" + "="*80)
#     print("EXAMPLE 4: AI Innovation Proposal")
#     print("="*80)
#     response4 = safaricom_finance_team.run(
#         """Create an AI fraud detection proposal for M-PESA:
        
#         1. POWERPOINT (MPESA_AI_Fraud_Detection.pptx):
#            - Problem statement: Current fraud losses
#            - Proposed AI solution architecture
#            - Expected ROI: 60% reduction in fraud, KES 500M annual savings
#            - Implementation timeline and milestones
           
#         2. EXCEL (AI_Fraud_ROI_Model.xlsx):
#            - Current fraud losses by type
#            - Projected savings over 3 years
#            - Implementation costs and ROI calculation"""
#     )
#     download_files_from_response(response4)

#     # Example 5: Treasury cash flow forecast
#     print("\n" + "="*80)
#     print("EXAMPLE 5: Treasury Cash Flow Forecast")
#     print("="*80)
#     response5 = safaricom_finance_team.run(
#         """Prepare Q1 2026 treasury documentation:
        
#         1. EXCEL (Q1_2026_Cashflow_Forecast.xlsx):
#            - Daily cash flow projections for January-March
#            - M-PESA float requirements: Daily average KES 25B
#            - Dividend payment: KES 40B in February
#            - Ethiopia capex: USD 25M monthly
#            - Include currency exposure analysis (KES/USD/ETB)"""
#     )
#     download_files_from_response(response5)
