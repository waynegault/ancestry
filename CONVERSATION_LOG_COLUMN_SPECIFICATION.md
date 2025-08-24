# Conversation Log Column Specification

## Overview
The `conversation_log` table tracks the latest message details for each conversation direction (IN/OUT) between the script and DNA matches. This document clarifies what should go in each key column.

## Column Specifications

### 1. `latest_message_content` (Text, Nullable)
**Purpose**: Stores the actual content of the latest message in this direction.

#### **Valid Content:**
- **For IN messages**: Raw message content received from users
- **For OUT messages**: The actual message text sent by the script
- **For Template Tracking**: Special tracking entries like "Template tracking: In_Tree-Initial_Confident"

#### **Content Rules:**
- **Truncated**: Limited to `message_truncation_length` (default: 1000 characters)
- **Raw Text**: No HTML encoding, preserve original formatting
- **Subject Included**: For email-style messages, include "Subject: ..." line
- **Placeholders Resolved**: For OUT messages, all {name}, {relationship} placeholders should be filled

#### **Examples:**
```
✅ VALID OUT message:
"Subject: DNA Match - Family Connection

Hi Emma,

I'm Wayne from Aberdeen, Scotland. Ancestry shows we're DNA matches and I believe you're my 4th cousin 1x removed.

Our connection appears to be:
Wayne Gault -> Emma (4th cousin 1x removed)

Best regards,
Wayne"

✅ VALID IN message:
"Hi Wayne, thanks for reaching out! I'd love to learn more about our family connection. My grandmother was from Scotland too."

✅ VALID Template Tracking:
"Template tracking: In_Tree-Initial_Confident"

❌ INVALID (unresolved placeholders):
"Hi {name}, I believe you're my {actual_relationship}"
```

---

### 2. `message_type_id` (Integer, Nullable, Foreign Key)
**Purpose**: Links to the MessageType table for script-sent messages only.

#### **When to Set:**
- **OUT messages**: Always set for script-sent messages
- **IN messages**: Always NULL (users don't send templated messages)
- **Template Tracking**: Set to 1 (default) or appropriate tracking ID

#### **Valid Values:**
Must correspond to existing `message_types.id` values:
- `1` = "In_Tree-Initial"
- `9` = "In_Tree-Initial_Short" 
- `10` = "Out_Tree-Initial_Short"
- `11` = "In_Tree-Initial_Confident"
- `12` = "Out_Tree-Initial_Exploratory"
- etc.

#### **Validation Rules:**
- **Required for OUT**: All script-sent messages must have valid message_type_id
- **NULL for IN**: User messages should never have message_type_id
- **Must Exist**: Foreign key must reference existing MessageType record

#### **Examples:**
```
✅ VALID:
OUT message with message_type_id = 11 (In_Tree-Initial_Confident)
IN message with message_type_id = NULL

❌ INVALID:
OUT message with message_type_id = NULL
IN message with message_type_id = 11
message_type_id = 999 (non-existent ID)
```

---

### 3. `script_message_status` (String, Nullable)
**Purpose**: Records the delivery/processing status of script actions.

#### **When to Set:**
- **OUT messages**: Always set for script-sent messages
- **IN messages**: Usually NULL, except for AI processing status
- **Template Tracking**: Set to tracking information

#### **Valid Status Values:**

##### **For OUT Messages (Script-Sent):**
- `"delivered OK"` - Message successfully sent via API
- `"typed (dry_run)"` - Message simulated in dry run mode
- `"skipped (interval)"` - Skipped due to minimum interval not met
- `"skipped (limit)"` - Skipped due to daily/run message limit
- `"skipped (filter)"` - Skipped by content filter
- `"failed (api_error)"` - API call failed
- `"failed (auth_error)"` - Authentication failed
- `"failed (rate_limit)"` - Rate limited by API

##### **For IN Messages (User-Sent):**
- `NULL` - Standard user message (most common)
- `"processed (ai_sentiment)"` - AI sentiment analysis completed
- `"processed (auto_reply)"` - Automated reply sent

##### **For Template Tracking:**
- `"TEMPLATE_SELECTED: {template_key} ({reason})"` - Template selection tracking
- Example: `"TEMPLATE_SELECTED: In_Tree-Initial_Confident (Confidence-based)"`

#### **Status Rules:**
- **Descriptive**: Should clearly indicate what happened
- **Consistent**: Use standardized status strings
- **Actionable**: Status should help with debugging/monitoring

#### **Examples:**
```
✅ VALID OUT statuses:
"delivered OK"
"typed (dry_run)"
"skipped (interval)"
"failed (rate_limit)"

✅ VALID IN statuses:
NULL
"processed (ai_sentiment)"

✅ VALID Template Tracking:
"TEMPLATE_SELECTED: In_Tree-Initial_Confident (Confidence-based)"

❌ INVALID:
"sent" (too vague)
"error" (not descriptive)
"ok" (not standardized)
```

---

## Data Consistency Rules

### **Cross-Column Validation:**
1. **OUT messages**: Must have both `message_type_id` AND `script_message_status`
2. **IN messages**: Should have `message_type_id = NULL` AND `script_message_status = NULL` (usually)
3. **Template Tracking**: Special case with tracking-specific values

### **Content-Type Alignment:**
- If `script_message_status = "typed (dry_run)"`, content should be actual message text
- If `script_message_status` contains "TEMPLATE_SELECTED", content should be tracking info
- Message content should match the template indicated by `message_type_id`

### **Relationship Consistency:**
- Template selection should match relationship distance in content
- "Confident" templates should not be used for 5th+ cousins
- "Exploratory" templates should be used for distant/unknown relationships

---

## Common Issues & Fixes

### **Issue 1: Template-Content Mismatch**
```
❌ PROBLEM:
message_type_id = 11 (In_Tree-Initial_Confident)
latest_message_content = "...you're my 6th cousin..."

✅ SOLUTION:
Use distance-aware template selection to prevent confident templates for distant relationships
```

### **Issue 2: Missing Status Information**
```
❌ PROBLEM:
OUT message with script_message_status = NULL

✅ SOLUTION:
Always set script_message_status for OUT messages, even if "unknown"
```

### **Issue 3: Inconsistent Tracking**
```
❌ PROBLEM:
Dual entries with different template information

✅ SOLUTION:
Use single source of truth for template selection and ensure consistency
```

---

## Monitoring & Validation Queries

### **Check for Invalid Combinations:**
```sql
-- OUT messages missing required fields
SELECT * FROM conversation_log 
WHERE direction = 'OUT' 
AND (message_type_id IS NULL OR script_message_status IS NULL);

-- IN messages with script fields set
SELECT * FROM conversation_log 
WHERE direction = 'IN' 
AND (message_type_id IS NOT NULL);

-- Template-content mismatches
SELECT cl.*, mt.type_name 
FROM conversation_log cl
JOIN message_types mt ON cl.message_type_id = mt.id
WHERE cl.direction = 'OUT'
AND mt.type_name LIKE '%Confident%'
AND cl.latest_message_content LIKE '%6th cousin%';
```

### **Data Quality Checks:**
```sql
-- Check for unresolved placeholders
SELECT * FROM conversation_log 
WHERE latest_message_content LIKE '%{%}%';

-- Check for missing template tracking
SELECT COUNT(*) FROM conversation_log 
WHERE script_message_status LIKE 'TEMPLATE_SELECTED%';
```

This specification ensures consistent, reliable data in the conversation_log table and prevents the mixups identified in the code review.
