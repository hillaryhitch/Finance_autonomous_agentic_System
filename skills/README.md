# Custom Skills Documentation for Safaricom Finance Agents

This directory contains custom skill documentation following Anthropic's skill structure format.

## Structure

Each skill is in its own folder with a `SKILL.md` file containing detailed domain knowledge:

```
skills/
├── safaricom_telco_expertise/
│   └── SKILL.md
├── mpesa_financial_services/
│   └── SKILL.md
└── financial_metrics/
    └── SKILL.md
```

## Available Skills

### 1. Safaricom Telecommunications Expertise
**Path:** `skills/safaricom_telco_expertise/SKILL.md`

Covers:
- Market position and competitive landscape
- Core telecommunications services (voice, data, SMS)
- Regional operations (Kenya, Ethiopia)
- Regulatory environment
- Strategic focus areas

### 2. M-PESA Financial Services
**Path:** `skills/mpesa_financial_services/SKILL.md`

Covers:
- Mobile money platform operations
- Financial products (lending, savings, payments)
- Ecosystem and integrations
- Regulatory compliance
- Risk management

### 3. Financial Metrics and Reporting
**Path:** `skills/financial_metrics/SKILL.md`

Covers:
- Key performance indicators (KPIs)
- IFRS reporting standards
- Segment performance metrics
- Currency exposure management
- Investment priorities

## Implementation Approach

**Important:** These SKILL.md files serve as **documentation** of domain knowledge. They are NOT directly used as Anthropic custom skills, as that would require registration with Anthropic's platform.

### How Knowledge is Provided to Agents

1. **Anthropic Official Skills** (Document Generation):
   - `pptx` - PowerPoint generation
   - `xlsx` - Excel spreadsheet generation
   - `docx` - Word document generation

2. **Domain Knowledge Sources**:
   - **Agent Instructions**: Context from SKILL.md files integrated into prompts
   - **Knowledge Base**: ChromaDB vector database with PDF embeddings
   - **Retrieval**: Semantic search over financial documents

3. **Architecture**:
   ```
   Agent Instructions (Domain Context)
         +
   Anthropic Official Skills (pptx/xlsx/docx)
         +
   Knowledge Base (ChromaDB + HuggingFace Embeddings)
         =
   Domain-Aware Financial Agents
   ```

## Benefits of This Approach

✅ **Clear Documentation**: SKILL.md files document domain knowledge  
✅ **Maintainable**: Easy to update and version control  
✅ **RAG-Ready**: Knowledge base provides retrieval-augmented generation  
✅ **Official Skills**: Reliable document generation with Anthropic's tools  
✅ **Scalable**: Can add more SKILL.md files as needed

## Future: True Custom Skills

To use these as registered Anthropic custom skills:
1. Register skills with Anthropic platform
2. Get assigned skill IDs
3. Reference in code as:
   ```python
   {"type": "custom", "skill_id": "your-registered-id", "version": "1.0"}
   ```

For now, the documented knowledge is integrated through instructions and the knowledge base, which provides equivalent functionality with more flexibility.
