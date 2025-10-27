# Intelligent DNA Match Messaging & Dialogue System - Vision Document

**Version**: 1.1
**Date**: October 21, 2025
**Status**: Approved - Ready for Implementation

---

## Executive Summary

Transform the Ancestry DNA match messaging system from a one-way broadcast tool into an **intelligent, conversational genealogical research assistant** that:

- Engages DNA matches in meaningful two-way dialogue about family connections
- Automatically researches and responds to genealogical questions using your family tree data
- Adapts messaging strategy based on relationship status, engagement patterns, and DNA data
- Creates actionable research tasks from incoming genealogical information
- Respects user preferences and manages conversation lifecycle intelligently

**Foundation**: Build on existing Actions 8 (Messaging) and 9 (Productive Processing) with enhanced AI-powered dialogue capabilities.

---

## Current State Assessment

### What's Working Well ✅

**Action 8 - Automated Messaging**:
- ✅ Differential messaging (in-tree vs out-of-tree templates)
- ✅ Message sequencing (Initial → Follow-Up → Final Reminder)
- ✅ Desist functionality (respects "do not contact" requests)
- ✅ Template selection based on confidence and A/B testing
- ✅ Dry run mode for safe testing
- ✅ 44,376+ messages successfully created and managed

**Action 9 - Productive Message Processing**:
- ✅ AI-powered message classification (PRODUCTIVE, DESIST, OTHER)
- ✅ Genealogical entity extraction (names, dates, places, relationships)
- ✅ Microsoft To-Do task creation for follow-up research
- ✅ Basic acknowledgment message sending
- ✅ DeepSeek AI integration with multi-provider abstraction

**Supporting Infrastructure**:
- ✅ Action 10: GEDCOM file analysis with relationship path calculation
- ✅ Action 11: API-based genealogical research
- ✅ Universal scoring system for person matching
- ✅ DNA ethnicity tracking in database
- ✅ Comprehensive conversation logging
- ✅ Rate limiting and error handling

### Current Limitations 🔧

1. **Limited Dialogue Intelligence**: Action 9 sends mostly generic acknowledgments, not substantive genealogical responses
2. **No Person Lookup Integration**: Doesn't use Action 10/11 to research people mentioned in messages
3. **Missing Relationship Context**: Responses don't include relationship paths or family details
4. **Unused DNA Data**: Ethnicity commonality not mentioned in messages
5. **Basic Tree Statistics**: Doesn't communicate "You're one of X matches in my tree"
6. **Static Follow-up Timing**: Uses fixed intervals, not engagement-based adaptation
7. **Limited Context Memory**: Doesn't maintain rich conversation state across exchanges

---

## Vision Statement

**Create an AI-powered genealogical research assistant that conducts intelligent, contextually-aware conversations with DNA matches, automatically researching family connections and providing substantive genealogical insights while respecting user preferences and managing conversation lifecycle.**

### Core Principles

1. **Intelligent & Helpful**: Provide real genealogical value in every response
2. **Contextually Aware**: Remember conversation history and adapt accordingly
3. **Respectful**: Honor do-not-contact preferences immediately
4. **Research-Driven**: Use Action 10/11 to look up people and relationships
5. **Data-Rich**: Leverage DNA ethnicity, tree statistics, and relationship paths
6. **Adaptive**: Adjust messaging when tree status changes (out-of-tree → in-tree)
7. **Task-Oriented**: Create actionable research tasks from new information

---

## Core Capabilities

### 1. Intelligent Initial Outreach

**Current**: Template-based messages with basic personalization
**Enhanced**:

- **In-Tree Matches**: Include specific relationship path
  - *"Hello cousin! According to my research, you're my 3rd cousin 1x removed. Our connection is: You → [parent] → [grandparent] → [common ancestor] ← [my ancestor] ← [my parent] ← Me. I'd love to verify this with you!"*

- **Out-of-Tree Matches**: Include tree statistics and DNA commonality
  - *"Hello! We're DNA matches (predicted 4th-6th cousin, sharing 45 cM). You're one of 2,500 DNA matches I have, but only 850 are in my tree so far. We both have significant Scottish ancestry (you: 35%, me: 42%). Would you like to explore our connection?"*

- **Dynamic Template Selection**: Use AI to select best template variant based on:
  - DNA confidence (cM shared, predicted relationship)
  - Tree status and relationship complexity
  - Last login activity (recent vs inactive users)
  - Ethnicity overlap

### 2. Conversational Dialogue Engine

**Current**: Generic acknowledgments for productive messages
**Enhanced**:

**Person Mention Detection & Lookup**:
- When recipient mentions: *"My grandfather was John Smith, born 1850 in Aberdeen"*
- System automatically:
  1. Extracts: John Smith, male, b. 1850, Aberdeen
  2. Searches using Action 10 (GEDCOM) and Action 11 (API)
  3. Generates response:
     - **Found**: *"I have John Smith (1850-1920) in my tree! He's my 3rd great-grandfather. His parents were William Smith and Mary Fraser. He married Jane Robertson in 1875. Is this the same person?"*
     - **Not Found**: *"I don't have a John Smith born 1850 in Aberdeen in my tree yet. My Aberdeen ancestors include the Gault, Fraser, and Milne families. Do you know if John Smith connected to any of these families?"*

**Relationship Path Responses**:
- For in-tree matches: Always include relationship path in responses
- For newly-added matches: *"Great news! I've added you to my tree. You're my 2nd cousin 2x removed through our common ancestor Margaret Fraser (1820-1890)."*

**DNA Ethnicity Commonality**:
- *"We both have strong connections to North East Scotland (you: 28%, me: 35%) and Galicia (southeast Poland/western Ukraine) (you: 12%, me: 15%). This suggests our common ancestor likely came from the Aberdeen/Banff area."*

**Tree Statistics Context**:
- *"You're one of 850 DNA matches I've successfully placed in my tree out of 2,500 total matches. Your connection helps fill in the Fraser family line."*

### 3. Adaptive Message Sequencing

**Current**: Fixed timing (Initial → Follow-Up → Final Reminder)
**Enhanced**:

**Engagement-Based Timing**:
- **Active Users** (logged in <7 days): Faster follow-up (7 days)
- **Moderate Users** (logged in 7-30 days): Standard follow-up (14 days)
- **Inactive Users** (logged in >30 days): Slower follow-up (21 days)
- **Never Logged In**: Single message only, no follow-ups

**Status Change Adaptation**:
- When out-of-tree match becomes in-tree:
  - Cancel pending out-of-tree follow-ups
  - Send new in-tree message: *"Update! I've found our connection. You're my [relationship] through [common ancestor]. Here's our relationship path: [path]"*

**Conversation Continuity**:
- If recipient replies between scheduled messages, cancel automated follow-ups
- Switch to conversational mode with AI-generated responses

### 4. Do-Not-Contact Management

**Current**: Desist acknowledgment message
**Enhanced**:

**Immediate Recognition**:
- Detect phrases: "stop", "unsubscribe", "not interested", "do not contact"
- Update Person.status to DESIST immediately
- Send polite acknowledgment: *"I completely understand and respect your preference. I've updated my records and won't contact you again. Thank you for letting me know!"*
- Cancel all pending messages
- Log reason for future analysis

**Proactive Preference Detection**:
- Detect soft signals: "I'm not interested in genealogy", "too busy right now"
- Mark for reduced contact frequency
- Offer opt-out: *"I understand you're busy. Would you prefer I don't send follow-up messages? Just let me know!"*

### 5. Genealogical Research Assistant

**Current**: Basic entity extraction and task creation
**Enhanced**:

**Multi-Person Lookup**:
- When recipient mentions multiple people: *"My grandparents were John Smith and Mary Jones, married in 1920"*
- System searches for both individuals
- Generates comprehensive response with all findings
- Creates relationship diagram if both found

**Source Citation**:
- Include source references in responses: *"According to my tree, John Smith (1850-1920) is documented in the 1881 Scotland Census (Banff, Aberdeenshire) and his death certificate (1920, Aberdeen)."*

**Research Suggestions**:
- *"Based on our connection, you might find these records helpful: [list of relevant Ancestry collections]. Would you like me to share specific records I've found for our common ancestors?"*

**Automated Task Creation**:
- Extract new genealogical information from messages
- Create Microsoft To-Do tasks with:
  - Person name and details
  - Source (message from [username])
  - Suggested action (verify, add to tree, find records)
  - Priority based on relationship closeness

### 6. Conversation State Management

**Enhanced Database Schema**:

```sql
-- New table: conversation_state
CREATE TABLE conversation_state (
    id INTEGER PRIMARY KEY,
    people_id INTEGER FOREIGN KEY,
    conversation_phase TEXT,  -- 'initial_outreach', 'active_dialogue', 'research_exchange', 'concluded'
    engagement_score INTEGER,  -- 0-100 based on response quality and frequency
    last_topic TEXT,  -- 'relationship_verification', 'person_lookup', 'dna_discussion'
    pending_questions TEXT,  -- JSON array of unanswered questions
    mentioned_people TEXT,  -- JSON array of people discussed
    shared_ancestors TEXT,  -- JSON array of confirmed common ancestors
    next_action TEXT,  -- 'await_reply', 'send_follow_up', 'research_needed', 'no_action'
    next_action_date DATETIME,
    created_at DATETIME,
    updated_at DATETIME
);
```

**Conversation Phases**:
1. **Initial Outreach**: First contact, awaiting response
2. **Active Dialogue**: Back-and-forth conversation in progress
3. **Research Exchange**: Sharing genealogical data and sources
4. **Concluded**: Conversation naturally ended or desist requested

---

## Technical Architecture

### Enhanced Action 9 - Intelligent Dialogue Engine

**New Functions**:

```python
def process_incoming_message(person: Person, message: ConversationLog) -> bool:
    """
    Main dialogue processing function.

    1. Classify message intent (PRODUCTIVE, DESIST, OTHER)
    2. Extract genealogical entities (people, dates, places)
    3. For each mentioned person:
       - Search using Action 10 (GEDCOM)
       - Search using Action 11 (API) if not found
       - Collect relationship paths and family details
    4. Generate AI response using:
       - Conversation history
       - Lookup results
       - DNA data (ethnicity, cM shared)
       - Tree statistics
       - Relationship paths
    5. Create MS To-Do tasks for new information
    6. Update conversation state
    7. Send response or schedule follow-up
    """

def lookup_mentioned_people(extracted_entities: dict) -> list[PersonLookupResult]:
    """
    Use Action 10/11 to find people mentioned in messages.
    Returns list of matches with relationship paths and family details.
    """

def generate_contextual_response(
    person: Person,
    conversation_history: list[ConversationLog],
    lookup_results: list[PersonLookupResult],
    dna_data: DnaMatch,
    tree_stats: dict
) -> str:
    """
    Generate AI-powered response using DeepSeek with:
    - Full conversation context
    - Genealogical lookup results
    - DNA ethnicity commonality
    - Tree statistics
    - Relationship paths
    """

def calculate_engagement_score(person: Person) -> int:
    """
    Calculate 0-100 engagement score based on:
    - Response frequency
    - Message quality (length, genealogical content)
    - Time to respond
    - Productive vs other sentiment ratio
    """

def determine_next_action(
    person: Person,
    conversation_state: ConversationState,
    latest_message: ConversationLog
) -> tuple[str, datetime]:
    """
    Determine next action and timing:
    - 'await_reply': Wait for response (no action)
    - 'send_follow_up': Send follow-up message
    - 'research_needed': Create task, await research
    - 'no_action': Conversation concluded
    """
```

### Enhanced Action 8 - Adaptive Messaging

**New Functions**:

```python
def select_template_with_ai(
    person: Person,
    dna_match: DnaMatch,
    family_tree: FamilyTree,
    tree_stats: dict
) -> str:
    """
    Use AI to select best template variant based on:
    - DNA confidence (cM, predicted relationship)
    - Tree status and relationship complexity
    - Last login activity
    - Ethnicity overlap
    - Previous message history
    """

def enrich_message_content(
    template: str,
    person: Person,
    dna_match: DnaMatch,
    family_tree: FamilyTree,
    tree_stats: dict,
    ethnicity_commonality: dict
) -> str:
    """
    Enrich template with:
    - Relationship paths (for in-tree)
    - Tree statistics (X of Y matches in tree)
    - DNA ethnicity commonality
    - Personalized research suggestions
    """

def calculate_tree_statistics() -> dict:
    """
    Calculate and cache:
    - Total DNA matches
    - Matches in tree vs out of tree
    - Matches by relationship tier (1st-2nd cousin, 3rd-4th, etc.)
    - Ethnicity region distribution
    """

def calculate_ethnicity_commonality(
    person_ethnicity: dict,
    my_ethnicity: dict
) -> dict:
    """
    Compare ethnicity regions and return:
    - Shared regions with percentages
    - Strongest commonalities
    - Geographic insights
    """
```

### AI Prompt Engineering

**New Prompts** (add to `ai_prompts.json`):

```json
{
  "genealogical_dialogue_response": {
    "system_prompt": "You are a genealogical research assistant helping Wayne connect with DNA matches...",
    "context_variables": [
      "conversation_history",
      "mentioned_people_lookup_results",
      "dna_ethnicity_commonality",
      "tree_statistics",
      "relationship_path"
    ]
  },
  "person_mention_extraction": {
    "system_prompt": "Extract all people mentioned in this genealogical conversation...",
    "output_format": "json_object"
  },
  "engagement_assessment": {
    "system_prompt": "Assess the engagement level and intent of this DNA match conversation...",
    "output_format": "json_object"
  }
}
```

---

## Implementation Phases

### Phase 1: Enhanced Message Content (Foundation)
**Goal**: Enrich existing messages with relationship paths, tree statistics, and DNA data

**Tasks**:
1. Implement `calculate_tree_statistics()` function
2. Implement `calculate_ethnicity_commonality()` function
3. Update Action 8 message templates to include:
   - Relationship paths for in-tree matches
   - Tree statistics for all messages
   - DNA ethnicity commonality for out-of-tree matches
4. Add database caching for tree statistics
5. Test with dry_run mode on 100 matches

**Success Criteria**:
- All in-tree messages include relationship paths
- All messages include tree statistics
- Out-of-tree messages mention ethnicity commonality when >10% overlap

### Phase 2: Person Lookup Integration (Intelligence)
**Goal**: Enable Action 9 to research people mentioned in messages

**Tasks**:
1. Implement `lookup_mentioned_people()` using Action 10/11
2. Enhance entity extraction to capture person details (name, birth year, place)
3. Create `PersonLookupResult` data structure
4. Integrate lookup results into AI response generation
5. Add conversation state tracking (new table)
6. Test with 20 real productive messages

**Success Criteria**:
- System successfully finds 80%+ of mentioned people in tree
- Responses include relationship paths for found people
- Responses acknowledge when people not found with helpful context

### Phase 3: Conversational Dialogue Engine (Transformation)
**Goal**: Transform Action 9 into intelligent dialogue system

**Tasks**:
1. Implement `generate_contextual_response()` with full context
2. Create new AI prompts for genealogical dialogue
3. Implement engagement scoring system
4. Add conversation phase tracking
5. Implement multi-person lookup and response generation
6. Test with 50 diverse message scenarios

**Success Criteria**:
- Responses are substantive and genealogically relevant
- System handles multi-person mentions correctly
- Engagement scores correlate with actual user engagement
- Conversation phases tracked accurately

### Phase 4: Adaptive Messaging & Status Changes (Optimization)
**Goal**: Make messaging system adaptive and intelligent

**Tasks**:
1. Implement engagement-based timing for follow-ups
2. Add status change detection (out-of-tree → in-tree)
3. Implement automatic message cancellation on status change
4. Create "update" message templates for status changes
5. Add conversation continuity (cancel automated messages on reply)
6. Test with 200 matches over 30-day period

**Success Criteria**:
- Follow-up timing adapts to user activity
- Status changes trigger appropriate messages
- No duplicate or conflicting messages sent
- Conversation flow feels natural

### Phase 5: Research Assistant Features (Advanced)
**Goal**: Add advanced genealogical research capabilities

**Tasks**:
1. Implement source citation in responses
2. Add research suggestion generation
3. Enhance MS To-Do task creation with priority and context
4. Implement relationship diagram generation
5. Add record sharing capabilities
6. Test with 10 active research conversations

**Success Criteria**:
- Responses include source citations when available
- Research suggestions are relevant and helpful
- Tasks created with appropriate priority and detail
- Users report high value from research assistance

### Phase 6: Production Deployment & Monitoring (Launch)
**Goal**: Deploy to production with monitoring and optimization

**Tasks**:
1. Comprehensive testing with Frances Milne account
2. A/B testing of message variants
3. Implement conversation analytics dashboard
4. Add engagement metrics tracking
5. Create feedback loop for continuous improvement
6. Full production deployment with Wayne's account

**Success Criteria**:
- Zero critical errors in production
- Response rate >15% (vs current ~5%)
- Engagement score >60 for active conversations
- User satisfaction feedback positive

### Phase 7: Local LLM Integration (Independence)
**Goal**: Migrate from DeepSeek to local LLM for privacy, cost savings, and independence

**Hardware Specifications**:
- **System**: Dell XPS 15 9520
- **CPU**: Intel Core i9-12900HK (14 cores, 20 threads, up to 5.0 GHz)
- **RAM**: 64GB DDR5
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop (4GB VRAM)
- **Storage**: NVMe SSD (fast model loading)

**Tasks**:
1. **Requirements Analysis**:
   - Document all AI capabilities needed (from Phases 1-6)
   - Identify token limits, response times, quality requirements
   - Determine acceptable inference speed (tokens/second)

2. **Model Evaluation**:
   - Research LLM options compatible with hardware:
     - **Llama 3.1 8B** (quantized Q4/Q5): Good general performance
     - **Mistral 7B** (quantized): Fast, efficient
     - **DeepSeek-Coder 6.7B** (quantized): Code/structured output
     - **Phi-3 Medium** (14B quantized): High quality, fits in RAM
   - Test inference frameworks:
     - **llama.cpp**: CPU/GPU hybrid, excellent quantization
     - **Ollama**: Easy setup, good performance
     - **vLLM**: Production-grade serving
     - **LM Studio**: User-friendly GUI

3. **Installation & Configuration**:
   - Install chosen framework (likely llama.cpp or Ollama)
   - Download and test candidate models
   - Benchmark performance (tokens/sec, quality)
   - Configure GPU offloading (4GB VRAM = ~10-15 layers)
   - Optimize for CPU inference (remaining layers)

4. **Provider Adapter**:
   - Create `LocalLLMProvider` class in `ai_interface.py`
   - Implement same interface as DeepSeek/Gemini providers
   - Add configuration to `.env`:
     ```
     AI_PROVIDER=local_llm
     LOCAL_LLM_MODEL_PATH=models/llama-3.1-8b-q5.gguf
     LOCAL_LLM_CONTEXT_SIZE=8192
     LOCAL_LLM_GPU_LAYERS=12
     LOCAL_LLM_THREADS=16
     ```

5. **Prompt Optimization**:
   - Test all AI prompts with local model
   - Adjust prompts for model-specific performance
   - Optimize token usage (local models may be slower)
   - Create model-specific prompt variants if needed

6. **Performance Testing**:
   - Benchmark response times vs DeepSeek
   - Test with real conversation scenarios
   - Measure quality of responses (human evaluation)
   - Identify any degradation in capabilities
   - Optimize inference parameters (temperature, top_p, etc.)

7. **Migration Strategy**:
   - Implement provider switching in config
   - Add fallback to DeepSeek if local model fails
   - Create performance monitoring
   - Gradual rollout (test with Frances first)

8. **Production Deployment**:
   - Deploy local LLM for all AI operations
   - Monitor performance and quality
   - Keep DeepSeek as backup provider
   - Document model management (updates, switching)

**Success Criteria**:
- Local LLM provides comparable quality to DeepSeek
- Response time <5 seconds for typical queries
- Zero API costs for AI operations
- Fallback to DeepSeek works seamlessly
- System runs reliably on laptop hardware

**Expected Performance**:
- **Llama 3.1 8B Q5**: ~15-25 tokens/sec (CPU+GPU hybrid)
- **Context Window**: 8K-32K tokens (depending on model)
- **Quality**: Comparable to DeepSeek for genealogical tasks

**Privacy Benefits**:
- All genealogical data stays local
- No external API calls for sensitive family information
- Complete control over model and data

---

## Success Metrics

### Quantitative Metrics
- **Response Rate**: Target 15%+ (vs current ~5%)
- **Engagement Score**: Average >60 for active conversations
- **Person Lookup Success**: 80%+ of mentioned people found
- **Task Completion**: 70%+ of created tasks completed
- **Conversation Duration**: Average 3+ message exchanges for productive conversations
- **Tree Growth**: 10%+ increase in matches added to tree

### Qualitative Metrics
- **Message Quality**: Responses are substantive and genealogically valuable
- **User Satisfaction**: Positive feedback from DNA matches
- **Research Value**: Conversations lead to new genealogical discoveries
- **Relationship Verification**: Increased confirmation of relationship paths

---

## Implementation Decisions (Approved)

### Testing Strategy ✅
- **Mode**: Use dry_run mode throughout development
- **Test Accounts**: Practice with real conversations between Wayne Gault and Frances Milne/Mchardy
- **Test Database**: Create isolated test database with just Frances for testing
- **Validation**: Test each phase before proceeding to next

### Implementation Approach ✅
- **Sequence**: Implement all phases sequentially (not parallel)
- **Scope**: Implement everything as described in this vision document
- **Commits**: Git commit at completion of each phase
- **Testing**: Comprehensive testing before phase completion

### AI Provider Strategy ✅

- **Current**: Continue with DeepSeek for Phases 1-6
- **Future**: Add Local LLM as Phase 7 (final phase)
- **Hardware**: Dell XPS 15 9520 (i9-12900HK, 64GB RAM, RTX 3050 Ti 4GB)
- **Rationale**: Determine LLM requirements first, then select compatible model

### Open Questions (To Be Determined During Implementation)

1. **Message Frequency**: Maximum messages per day in production?
   - Suggested: 50-100 per day to avoid overwhelming recipients

2. **Engagement Thresholds**: When to stop messaging?
   - After 3 unanswered messages (current)?
   - Based on engagement score <20?
   - After X days of inactivity?

3. **Tree Statistics**: Specific numbers or vague?
   - "You're one of 850 matches in my tree" (specific)
   - "You're one of many matches I've placed in my tree" (vague)

4. **Ethnicity Sharing**: Detail level?
   - High-level regions only (Scotland, Spain)
   - Detailed sub-regions (North East Scotland, Galicia)
   - Percentage ranges or exact percentages?

5. **Research Sharing**: Proactive or on-request?
   - Wait for request
   - Offer to share in initial message
   - Automatically attach relevant records

6. **Conversation Archival**: When to conclude?
   - After 30 days of inactivity?
   - After explicit "goodbye" message?
   - Never (keep all conversations active)?

7. **Priority Matching**: Prioritization strategy?
   - Close DNA matches (>100 cM) first?
   - In-tree matches first?
   - Recent logins first?
   - Random order?

---

## Next Steps

1. ✅ **Vision Document Approved** - Ready for implementation
2. ✅ **Implementation Strategy Confirmed** - Sequential phases, dry_run mode, test database
3. 🔄 **Create Augment Task List** - Break down all phases into detailed tasks
4. 🔄 **Begin Phase 1** - Enhanced message content (foundation)

---

**Document Status**: ✅ Approved - Implementation Ready
**Estimated Implementation Time**: 8-10 weeks (7 phases including Local LLM)
**Dependencies**: Existing Actions 8, 9, 10, 11, AI interface, database schema
**Test Strategy**: Dry-run mode, Frances Milne test account, isolated test database



