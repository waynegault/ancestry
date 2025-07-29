# COMPREHENSIVE TESTING PROGRAM
## Safe Testing Protocol for Ancestry Research Automation

### 🎯 **TESTING OBJECTIVES**

**Primary Goals:**
- a. Understand which DNA matches are in the Gault family tree
- b. Maintain complete record of messages sent/received
- c. Identify matches to exclude from future communication
- d. Determine appropriate messages for each match type
- e. Validate AI interpretation and tree update capabilities

**Safety Constraint:**
- **ONLY** send test messages to: **Frances McHardy (nee Milne)**
- **NEVER** send test content to other DNA matches

---

### 📋 **PHASE 1: DATABASE ANALYSIS & CURRENT STATE**

#### **Step 1.1: DNA Match Inventory**
```sql
-- Query to understand current DNA match distribution
SELECT 
    COUNT(*) as total_matches,
    SUM(CASE WHEN in_my_tree = 1 THEN 1 ELSE 0 END) as in_tree_count,
    SUM(CASE WHEN in_my_tree = 0 THEN 1 ELSE 0 END) as out_tree_count,
    COUNT(DISTINCT status) as status_types
FROM people 
WHERE deleted_at IS NULL;

-- Detailed breakdown by status
SELECT status, COUNT(*) as count 
FROM people 
WHERE deleted_at IS NULL 
GROUP BY status;
```

#### **Step 1.2: Tree Placement Analysis**
```sql
-- Analyze Gault family tree connections
SELECT 
    ft.actual_relationship,
    COUNT(*) as match_count,
    AVG(dm.cM_DNA) as avg_cm
FROM family_tree ft
JOIN people p ON ft.people_id = p.id
JOIN dna_match dm ON dm.people_id = p.id
WHERE p.deleted_at IS NULL
GROUP BY ft.actual_relationship
ORDER BY match_count DESC;
```

#### **Step 1.3: Communication History Analysis**
```sql
-- Message exchange summary
SELECT 
    p.username,
    p.status,
    p.in_my_tree,
    COUNT(DISTINCT cl.conversation_id) as conversation_count,
    MAX(cl.latest_timestamp) as last_contact,
    cl.ai_sentiment
FROM people p
LEFT JOIN conversation_log cl ON p.id = cl.people_id
WHERE p.deleted_at IS NULL
GROUP BY p.id, p.username, p.status, p.in_my_tree, cl.ai_sentiment
ORDER BY last_contact DESC;
```

---

### 📋 **PHASE 2: SAFE TESTING SETUP**

#### **Step 2.1: Create Test Environment**
```python
# Add to config_schema.py
@dataclass
class TestingConfig:
    """Testing configuration to ensure safe operation."""
    
    # Safe testing mode
    testing_mode: bool = True
    test_recipient_only: bool = True
    
    # Approved test recipient
    test_recipient_name: str = "Frances McHardy"
    test_recipient_alt_name: str = "Frances Milne"
    test_recipient_username_patterns: List[str] = field(default_factory=lambda: [
        "frances", "fran", "mchardy", "milne"
    ])
    
    # Safety checks
    require_explicit_approval: bool = True
    log_all_test_actions: bool = True
```

#### **Step 2.2: Implement Safety Guards**
```python
def is_safe_test_recipient(person: Person) -> bool:
    """Verify if person is approved for testing."""
    if not config_schema.testing.testing_mode:
        return False
        
    username_lower = person.username.lower()
    approved_patterns = config_schema.testing.test_recipient_username_patterns
    
    return any(pattern in username_lower for pattern in approved_patterns)

def safe_message_send(person: Person, message: str) -> bool:
    """Send message only if recipient is approved for testing."""
    if not is_safe_test_recipient(person):
        logger.warning(f"BLOCKED: Attempted to send test message to {person.username}")
        return False
        
    logger.info(f"SAFE: Sending test message to approved recipient {person.username}")
    return send_message_api(person, message)
```

---

### 📋 **PHASE 3: SYSTEMATIC TESTING PROTOCOL**

#### **Step 3.1: Message Template Testing**
1. **Identify Frances McHardy in Database**
   ```python
   # Find Frances in the database
   frances = session.query(Person).filter(
       Person.username.ilike('%frances%'),
       Person.deleted_at.is_(None)
   ).first()
   ```

2. **Test Each Message Type**
   - `In_Tree-Initial`: If Frances is in tree
   - `Out_Tree-Initial`: If Frances is not in tree
   - `Follow_Up`: Test follow-up logic
   - `Final_Reminder`: Test reminder logic
   - `User_Requested_Desist`: Test desist acknowledgment

#### **Step 3.2: AI Processing Validation**
1. **Create Test Messages**
   ```python
   test_messages = [
       "Thank you for reaching out! I believe we're related through the Gault line...",
       "Please don't contact me again about DNA matches.",
       "I have information about John Gault born in 1850...",
       "My grandmother was Mary Milne from Aberdeen..."
   ]
   ```

2. **Test AI Classification**
   ```python
   for message in test_messages:
       classification = classify_message_intent(message, session_manager)
       extracted_data = extract_genealogical_entities(message, session_manager)
       print(f"Message: {message[:50]}...")
       print(f"Classification: {classification}")
       print(f"Extracted: {extracted_data}")
   ```

#### **Step 3.3: Tree Integration Testing**
1. **GEDCOM Analysis**
   ```python
   # Test GEDCOM search for Frances
   search_results = search_gedcom_persons(
       search_criteria={"first_name": "Frances", "surname": "McHardy"},
       max_results=10
   )
   ```

2. **Relationship Path Calculation**
   ```python
   # Test relationship path finding
   if frances_in_tree:
       path = calculate_relationship_path(user_id, frances.cfpid)
       print(f"Relationship to Frances: {path}")
   ```

---

### 📋 **PHASE 4: COMPREHENSIVE VALIDATION**

#### **Step 4.1: End-to-End Workflow Test**
1. **Controlled Full Workflow**
   - Run Action 7 (Search Inbox) - READ ONLY
   - Run Action 9 (Process Messages) - ANALYSIS ONLY
   - Run Action 8 (Send Messages) - FRANCES ONLY

2. **Validation Checkpoints**
   - Verify no messages sent to non-approved recipients
   - Confirm all AI classifications are logged
   - Validate database updates are accurate
   - Check message template rendering

#### **Step 4.2: Data Integrity Verification**
```python
def validate_test_results():
    """Comprehensive validation of test results."""
    
    # Check no unauthorized messages sent
    unauthorized_messages = session.query(ConversationLog).filter(
        ConversationLog.direction == MessageDirectionEnum.OUT,
        ConversationLog.latest_timestamp >= test_start_time,
        ~ConversationLog.people_id.in_(approved_recipient_ids)
    ).count()
    
    assert unauthorized_messages == 0, "Unauthorized messages detected!"
    
    # Verify AI processing accuracy
    processed_messages = session.query(ConversationLog).filter(
        ConversationLog.ai_sentiment.isnot(None),
        ConversationLog.latest_timestamp >= test_start_time
    ).all()
    
    for msg in processed_messages:
        assert msg.ai_sentiment in EXPECTED_INTENT_CATEGORIES
    
    print("✅ All validation checks passed!")
```

---

### 📋 **PHASE 5: PRODUCTION READINESS**

#### **Step 5.1: Final Safety Checklist**
- [ ] All test messages sent only to Frances McHardy
- [ ] AI classification accuracy validated
- [ ] Database integrity confirmed
- [ ] Message templates render correctly
- [ ] Tree placement logic verified
- [ ] Error handling tested
- [ ] Rate limiting respected
- [ ] Backup created before testing

#### **Step 5.2: Production Configuration**
```python
# Switch to production mode
config_schema.testing.testing_mode = False
config_schema.testing.test_recipient_only = False
config_schema.app_mode = "production"
```

---

### 🚨 **EMERGENCY PROCEDURES**

#### **If Unauthorized Message Sent:**
1. Immediately stop all automation
2. Check conversation_log for unauthorized entries
3. Send manual apology if needed
4. Review and fix safety guards
5. Re-run validation tests

#### **Database Rollback:**
```python
# Restore from backup if needed
restore_db_actn(session_manager, "pre_test_backup")
```

---

### 📊 **SUCCESS METRICS**

**Testing Complete When:**
- ✅ 100% of test messages sent only to approved recipient
- ✅ AI classification accuracy > 95%
- ✅ Tree placement logic validated
- ✅ Message template rendering confirmed
- ✅ Database integrity maintained
- ✅ All safety guards functional
- ✅ Production readiness confirmed

**Ready for Production When:**
- All testing phases completed successfully
- Safety protocols validated
- Backup procedures tested
- Emergency procedures documented
- Team trained on system operation
