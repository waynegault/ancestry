# Ancestry Research Automation - Technical Documentation

## System Architecture Overview

This is a comprehensive genealogical research automation system built in Python 3.8+ that integrates with Ancestry.com's web interface and APIs to automate DNA match analysis, message processing, and research task generation.

### Core Technology Stack
- **Python 3.8+** with type hints and modern async patterns
- **SQLAlchemy ORM** with SQLite database for local data storage
- **Selenium WebDriver** for web automation and session management
- **AI Integration** (DeepSeek/Google Gemini) for content analysis
- **Microsoft Graph API** for task management integration
- **Cryptography (Fernet)** for secure credential storage

## Application Structure

### Core Packages

#### `core/` Package - Infrastructure Layer
- **`session_manager.py`** - Manages browser sessions, cookies, and authentication state
- **`database_manager.py`** - SQLAlchemy session management and connection pooling
- **`browser_manager.py`** - Selenium WebDriver lifecycle and configuration
- **`api_manager.py`** - HTTP request handling with CSRF token management
- **`error_handling.py`** - Comprehensive error handling with circuit breakers and retry logic

#### `config/` Package - Configuration Layer
- **`config_manager.py`** - Environment-based configuration loading with validation
- **`config_schema.py`** - Pydantic-style dataclass schemas for type-safe configuration
- **`credential_manager.py`** - Encrypted credential storage using system keyring

### Action Modules - Business Logic Layer

#### `action6_gather.py` - DNA Match Collection
**Purpose**: Systematically collects DNA match data from Ancestry.com
**API Endpoints Used**:
- `GET /discoveryui-matches/api/matches` - Fetches paginated DNA match lists
- `POST /discoveryui-matches/api/matches/{match_id}/intree` - Checks tree linkage status

**Data Flow**:
1. Authenticates via Selenium browser session
2. Extracts cookies for API authentication
3. Paginates through match results (configurable page limits)
4. Compares with existing database records
5. Fetches additional details via concurrent API calls
6. Performs bulk database updates using SQLAlchemy

**Key Features**:
- Adaptive rate limiting (0.1-2.0 RPS based on API response patterns)
- Circuit breaker pattern for failure tolerance
- Progress tracking with tqdm integration
- Concurrent API fetching using ThreadPoolExecutor

#### `action7_inbox.py` - Message Processing
**Purpose**: Processes Ancestry inbox messages with AI-powered classification
**API Endpoints Used**:
- `GET /messaging/api/conversations` - Fetches conversation list
- Individual conversation detail endpoints for message content

**AI Integration**:
- Uses `ai_interface.py` to classify message intent (PRODUCTIVE, DESIST, UNINTERESTING)
- Extracts genealogical entities (names, dates, locations) from message content
- Stores AI analysis results in `ConversationLog` table

**Processing Logic**:
1. Fetches conversations in configurable batches
2. Compares timestamps with database to identify new messages
3. Applies AI classification to determine message value
4. Updates database with conversation status and AI insights

#### `action8_messaging.py` - Automated Messaging
**Purpose**: Sends personalized messages to DNA matches
**API Endpoints Used**:
- `POST /app-api/express/v2/conversations/message` - Send new message
- `POST /app-api/express/v2/conversations/{conv_id}` - Reply to existing conversation

**Message Personalization**:
- Uses `message_personalization.py` with 20+ dynamic functions
- Integrates genealogical data (names, dates, locations, relationships)
- 6 enhanced message templates with placeholder substitution
- Graceful fallbacks for missing data

**Safety Features**:
- Dry-run mode for testing without sending
- Duplicate message prevention
- Rate limiting and error recovery
- Message truncation and validation

#### `action9_process_productive.py` - Task Generation
**Purpose**: Converts productive conversations into actionable research tasks
**Integration Points**:
- Microsoft Graph API for To-Do task creation
- `genealogical_task_templates.py` for specialized research templates
- AI analysis for extracting actionable information

**Task Templates** (8 specialized types):
1. Vital Records Research
2. DNA Analysis Tasks
3. Immigration Research
4. Census Record Analysis
5. Military Service Research
6. Occupation Research
7. Location-Based Research
8. Family Relationship Verification

#### `action10.py` - GEDCOM Analysis (Local)
**Purpose**: Analyzes local GEDCOM files for research opportunities
**Features**:
- Hardcoded filtering with OR logic for individual selection
- Scoring algorithm based on genealogical criteria
- Relationship path calculation using cached GEDCOM data
- Top candidate identification and ranking

#### `action11.py` - API Report (Online)
**Purpose**: Compares family tree data with Ancestry's online database
**API Endpoints Used**:
- `GET /api/person-picker/suggest/{tree_id}` - Person search
- `GET /family-tree/person/facts/user/{profile_id}/tree/{tree_id}/person/{person_id}` - Person facts
- `GET /family-tree/person/tree/{tree_id}/person/{person_id}/getladder` - Relationship paths

## Database Schema

### Core Tables

#### `people` Table - Central Entity
```sql
- id (Primary Key)
- uuid (Ancestry DNA Sample ID)
- profile_id (Ancestry User Profile ID)
- display_name, first_name, last_name
- in_my_tree (Boolean flag)
- contactable (Boolean flag)
- last_logged_in (Timestamp)
- administrator_profile_id (For managed kits)
```

#### `dna_matches` Table - Genetic Information
```sql
- person_id (Foreign Key to people)
- shared_dna_cm (Centimorgans)
- estimated_relationship
- confidence_level
- match_date (When match was discovered)
- notes (User annotations)
```

#### `family_trees` Table - Genealogical Context
```sql
- person_id (Foreign Key to people)
- person_name_in_tree
- actual_relationship (Calculated relationship)
- relationship_path (Text description)
- facts_link, view_in_tree_link (URLs)
```

#### `conversation_logs` Table - Communication Tracking
```sql
- person_id (Foreign Key to people)
- conversation_id (Ancestry conversation ID)
- direction (IN/OUT enum)
- latest_message_content (Truncated)
- latest_timestamp
- ai_sentiment (AI classification)
- message_type_id (Template used)
- script_message_status (Delivery status)
```

## API Integration Details

### Authentication Flow
1. **Browser Login**: Selenium authenticates via web interface
2. **Cookie Extraction**: Transfers session cookies to requests session
3. **CSRF Token**: Retrieves token from `/discoveryui-matches/parents/api/csrfToken`
4. **API Requests**: Uses cookies + CSRF token for authenticated calls

### Required Cookies
- **`ANCSESSIONID`**: Primary session identifier
- **`SecureATT`**: Security authentication token
- **`OptanonConsent`**: Cookie consent (for DNA operations)
- **`trees`**: Tree access permissions

### Rate Limiting Strategy
```python
# Conservative settings post-hardening
requests_per_second: 0.5
initial_delay: 2.0
retry_backoff_factor: 4.0
failure_threshold: 10
burst_limit: 3
```

## AI Integration Architecture

### Supported Providers
- **DeepSeek**: Primary AI provider for genealogical analysis
- **Google Gemini**: Alternative provider with fallback support

### AI Functions
1. **Message Classification**: Categorizes inbox messages by research value
2. **Entity Extraction**: Identifies genealogical data (names, dates, places)
3. **Task Generation**: Creates specific research tasks from conversation content
4. **GEDCOM Analysis**: Identifies gaps and conflicts in family tree data

### Prompt Engineering
- Specialized prompts in `ai_prompts.json` for genealogical contexts
- Real genealogical examples for improved accuracy
- Structured JSON output aligned with Pydantic models

## GEDCOM AI Features (Phase 12)

### `gedcom_intelligence.py` - Family Tree Analysis
- Analyzes family completeness and identifies missing information
- Detects date inconsistencies and relationship conflicts
- Identifies location patterns and research clusters
- Generates AI-powered insights and recommendations

### `dna_gedcom_crossref.py` - DNA Integration
- Cross-references DNA matches with GEDCOM individuals
- Identifies potential family tree connections
- Suggests verification tasks for genetic evidence
- Provides confidence scoring for matches

### `research_prioritization.py` - Task Optimization
- Prioritizes research tasks based on multiple factors
- Creates location-based research clusters
- Estimates effort and success probability
- Generates comprehensive research plans

## Security Architecture

### Credential Management
- **Fernet Encryption**: Industry-standard symmetric encryption
- **System Keyring**: OS-level master key storage
- **File Permissions**: Restrictive access (0o600) for credential files
- **Migration Support**: Safe transition from plaintext to encrypted storage

### Data Protection
- **Local Storage**: All data stored locally (no cloud dependencies)
- **Encrypted Backups**: Optional backup encryption
- **Session Security**: Secure cookie handling and CSRF protection

## Performance Optimization

### Adaptive Rate Limiting
```python
class AdaptiveRateLimiter:
    - Monitors API response patterns
    - Adjusts RPS based on success rates
    - Implements exponential backoff for 429 errors
    - Provides performance analytics
```

### Caching Strategy
- **GEDCOM Caching**: Preprocessed family tree data
- **API Response Caching**: Reduces redundant requests
- **Database Connection Pooling**: Optimized SQLAlchemy sessions

### Error Handling
- **Circuit Breaker Pattern**: Prevents cascade failures
- **Retry Logic**: Exponential backoff with jitter
- **Graceful Degradation**: System continues with reduced functionality
- **Comprehensive Logging**: Detailed error tracking and recovery

## Testing Framework

### Test Coverage
- **393 Tests** across 46 modules
- **100% Success Rate** maintained
- **Multiple Test Categories**: Initialization, functionality, error handling, integration

### Test Types
1. **Unit Tests**: Individual function validation
2. **Integration Tests**: Cross-component interaction
3. **Performance Tests**: Timing and efficiency validation
4. **Error Handling Tests**: Failure scenario coverage

## Configuration Management

### Environment Variables
```bash
# Core settings
MAX_PAGES=1                    # Limit page processing
MAX_INBOX=5                    # Inbox conversation limit
MAX_PRODUCTIVE_TO_PROCESS=5    # Task generation limit
BATCH_SIZE=5                   # Processing batch size

# AI Configuration
AI_PROVIDER=deepseek           # AI service selection
DEEPSEEK_API_KEY=sk-xxx        # API authentication

# Database
DATABASE_FILE=ancestry.db      # SQLite database path
GEDCOM_FILE_PATH=family.ged    # GEDCOM file location
```

### Configuration Schema Validation
- Type-safe configuration using dataclasses
- Environment variable precedence
- Default value fallbacks
- Runtime validation and error reporting

## Microsoft Graph Integration

### Task Management Workflow
```python
# ms_graph_utils.py integration
- MSAL authentication with persistent token caching
- To-Do list management and task creation
- Automated task categorization and priority assignment
- Integration with genealogical research templates
```

### Task Creation Process
1. **Authentication**: MSAL handles OAuth2 flow with token refresh
2. **List Management**: Creates/manages "Ancestry Tasks" list
3. **Task Generation**: Converts AI analysis into actionable tasks
4. **Priority Assignment**: Based on research value and success probability

## Utility Modules

### `api_utils.py` - Ancestry API Abstraction
**Functions**:
- `parse_person_details()` - Extracts person data from API responses
- `format_relationship_paths()` - Processes getladder API responses
- `call_suggest_api()` - Person search functionality
- `call_facts_api()` - Retrieves person facts and details
- `send_message_api()` - Message sending with error handling

### `gedcom_search_utils.py` - Family Tree Processing
**Capabilities**:
- GEDCOM file parsing and caching
- Person search with fuzzy matching
- Relationship path calculation
- Data structure optimization for large family trees

### `message_personalization.py` - Dynamic Content Generation
**Personalization Functions** (20+ available):
- Name formatting and relationship calculation
- Date and location processing
- Research question generation
- Template placeholder substitution

### `genealogical_task_templates.py` - Research Task Generation
**Template Categories**:
1. **Vital Records**: Birth/death certificate research
2. **DNA Analysis**: Genetic evidence evaluation
3. **Immigration**: Ship manifests and naturalization records
4. **Census**: Population schedule analysis
5. **Military**: Service record research
6. **Occupation**: Professional history investigation
7. **Location**: Geographic research clusters
8. **Relationship**: Family connection verification

## Error Handling and Monitoring

### Circuit Breaker Implementation
```python
@circuit_breaker(failure_threshold=10, recovery_timeout=300)
def api_operation():
    # Automatic failure detection and recovery
    # Prevents cascade failures across system
```

### Monitoring and Alerting
- **Failure Pattern Detection**: Alerts at 50% of failure threshold
- **Critical Warnings**: Escalated alerts at 80% threshold
- **Performance Dashboards**: Real-time system health monitoring
- **Configuration Validation**: Startup validation of all settings

### Logging Architecture
- **Structured Logging**: Consistent format across all modules
- **Log Level Management**: DEBUG/INFO/WARNING/ERROR with dynamic control
- **Context Preservation**: Error context tracking through call stacks
- **Performance Metrics**: Timing and efficiency logging

## Development and Deployment

### Code Quality Standards
- **Type Safety**: Full type hints throughout codebase
- **Error Handling**: Comprehensive exception handling with detailed logging
- **Testing**: 10-15 test categories per module with 95%+ coverage
- **Documentation**: Clear docstrings and usage examples

### Package Structure Benefits
- **Dual-Mode Operation**: Modules work individually and as packages
- **Import Standardization**: Consistent import patterns with fallbacks
- **Path Resolution**: Automatic parent directory path insertion
- **Dependency Injection**: Modern DI patterns reduce coupling

### Performance Characteristics
- **Memory Usage**: Optimized data structures with intelligent caching
- **Database Performance**: Indexed SQLite with connection pooling
- **API Efficiency**: Dynamic throttling prevents rate limit restrictions
- **Error Recovery**: Automatic retry with exponential backoff

## Integration Points

### External Service Dependencies
1. **Ancestry.com**: Primary data source via web scraping and API
2. **AI Providers**: DeepSeek or Google Gemini for content analysis
3. **Microsoft Graph**: Task management and productivity integration
4. **System Keyring**: Secure credential storage

### Data Flow Architecture
```
Ancestry.com → Browser Session → Cookie Extraction → API Calls →
Database Storage → AI Analysis → Task Generation → Microsoft To-Do
```

### Backup and Recovery
- **Database Backups**: Automated with compression and encryption options
- **Configuration Backups**: Environment and credential preservation
- **Recovery Procedures**: Automated restore with validation
- **Data Migration**: Safe upgrade paths between versions

## Troubleshooting Guide

### Common Issues and Solutions
1. **Authentication Failures**: Session timeout, credential validation
2. **Rate Limiting**: 429 errors, adaptive rate limiter adjustment
3. **Database Errors**: Connection issues, disk space, permissions
4. **AI API Failures**: Provider outages, quota limits, fallback strategies
5. **CSRF Token Issues**: Session expiration, token refresh procedures

### Diagnostic Tools
- **Test Suites**: Individual module testing for component validation
- **Performance Dashboard**: Real-time system health monitoring
- **Log Analysis**: Structured logging with searchable error patterns
- **Configuration Validation**: Startup checks for all required settings

### Maintenance Procedures
- **Regular Testing**: Automated test execution with `run_all_tests.py`
- **Performance Monitoring**: Adaptive system optimization tracking
- **Database Maintenance**: Backup rotation and cleanup procedures
- **Security Updates**: Credential rotation and encryption key management

This comprehensive technical documentation provides complete coverage of the Ancestry Research Automation system architecture, implementation details, and operational procedures.
