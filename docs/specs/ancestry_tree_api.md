# Ancestry.com Family Tree Write APIs

> **Status**: Discovered via browser DevTools inspection (December 2025)
> **Purpose**: Enable Phase 8 - Tree Update Automation

## Overview

These APIs allow programmatic modification of Ancestry.com family trees. They require authenticated session cookies from an active browser session.

---

## Authentication

All requests require these cookies from an authenticated Ancestry session:
- `ATT` - Primary auth token
- `SecureATT` - JWT with session details
- `ANCATT` - Secondary auth token
- `ANCSESSIONID` - Session identifier

**Required Headers:**
```
Content-Type: application/json
Accept: application/json
```

---

## API Endpoints

### Base URL Pattern
```
https://www.ancestry.co.uk/family-tree/person/
  addedit/user/{userId}/tree/{treeId}/person/{personId}/...
```

**Path Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `userId` | User GUID | `07bdd45e-0006-0000-0000-000000000000` |
| `treeId` | Tree ID | `175946702` |
| `personId` | Person ID | `102281560244` |

---

## 1. Update Person (Quick Edit)

Updates core person facts: name, birth, death, gender, living status.

**Endpoint:**
```
POST /family-tree/person/addedit/user/{userId}/tree/{treeId}/person/{personId}/updatePerson
```

**Request Body:**
```json
{
  "person": {
    "personId": "102281560244",
    "treeId": "175946702",
    "userId": "07bdd45e-0006-0000-0000-000000000000",
    "gender": "Male"
  },
  "values": {
    "fname": "John",
    "mname": "William",
    "lname": "Smith",
    "sufname": "Jr",
    "bdate": "1 Jan 1900",
    "bplace": "London, England",
    "ddate": "15 Mar 1975",
    "dplace": "Manchester, England",
    "statusRadio": "Deceased",
    "genderRadio": "Male"
  }
}
```

**Field Mapping to FactTypeEnum:**
| Field | Maps To | Description |
|-------|---------|-------------|
| `bdate` | `BIRTH` | Birth date (free text) |
| `bplace` | `BIRTH` | Birth place |
| `ddate` | `DEATH` | Death date (free text) |
| `dplace` | `DEATH` | Death place |
| `fname/mname/lname/sufname` | - | Name components |
| `genderRadio` | - | `Male` / `Female` / `Person` (unknown) |
| `statusRadio` | - | `Living` / `Deceased` |

**Response:** JSON with updated person data

---

## 2. Add Fact (Extended Facts)

Adds any fact type to a person (residence, occupation, military, etc.).

**Get Form (for new fact):**
```
GET /family-tree/person/factedit/user/{userId}/tree/{treeId}/person/{personId}/assertion/0?eventType={type}
```

**Save Fact:**
```
POST /family-tree/person/factedit/user/{userId}/tree/{treeId}/person/{personId}/assertion/0/save
```

**Request Body:**
```json
{
  "assertionId": "0",
  "eventType": "death",
  "date": "1999",
  "location": "London, England",
  "description": "Additional details",
  "sourceInfo": null
}
```

**Event Types (maps to FactTypeEnum):**
| eventType | FactTypeEnum | Description |
|-----------|--------------|-------------|
| `birth` | `BIRTH` | Birth event |
| `death` | `DEATH` | Death event |
| `marriage` | `MARRIAGE` | Marriage event |
| `residence` | `LOCATION` | Residence/address |
| `occupation` | `OTHER` | Occupation/profession |
| `military` | `OTHER` | Military service |
| `immigration` | `OTHER` | Immigration event |
| `emigration` | `OTHER` | Emigration event |
| `naturalization` | `OTHER` | Naturalization |
| `census` | `OTHER` | Census record |

**Notes:**
- `assertionId: "0"` = create new fact
- Use existing assertionId to update a fact

---

## 3. Add Person with Relationship

Creates a new person and establishes a relationship to an existing person.

**Endpoint:**
```
POST /family-tree/person/addedit/user/{userId}/tree/{treeId}/person/{personId}/addperson
```

### 3a. Add Spouse

```json
{
  "addTarget": null,
  "person": {
    "personId": "102281560244",
    "treeId": "175946702",
    "userId": "07bdd45e-0006-0000-0000-000000000000",
    "gender": "Male"
  },
  "type": "Spouse",
  "values": {
    "fname": "Jane",
    "lname": "Doe",
    "sufname": "",
    "genderRadio": "Female",
    "statusRadio": "Living",
    "spousalRelationship": "Spouse",
    "bdate": "",
    "bplace": "",
    "isAlternateParent": false,
    "nameId": "505366695093",
    "genderId": "505366695099"
  }
}
```

### 3b. Add Child

```json
{
  "addTarget": null,
  "person": {
    "personId": "102281560244",
    "treeId": "175946702",
    "userId": "07bdd45e-0006-0000-0000-000000000000",
    "gender": "Male"
  },
  "type": "Child",
  "values": {
    "fname": "John",
    "lname": "Smith",
    "sufname": "",
    "genderRadio": "Male",
    "statusRadio": "Living",
    "bdate": "",
    "bplace": "",
    "parentSet": {
      "fatherId": "102281560244",
      "motherId": ""
    },
    "isAlternateParent": false,
    "nameId": "505366695093",
    "genderId": "505366695099"
  }
}
```

### 3c. Add Parent (inferred)

Based on patterns, likely:
```json
{
  "type": "Father",
  "values": {...}
}
```
or
```json
{
  "type": "Mother",
  "values": {...}
}
```

### 3d. Link Existing Person as Relationship

Instead of creating a new person, link an **existing person** already in the tree:

```json
{
  "person": {
    "personId": "102281560836",
    "treeId": "175946702",
    "userId": "07bdd45e-0006-0000-0000-000000000000"
  },
  "type": "Child",
  "values": {
    "apmFindExistingPerson": {
      "name": "Hamish Stuart Cruickshank",
      "birth": "1940",
      "death": "2005",
      "PID": 102601312241,
      "genderIconType": "Male"
    },
    "parentSet": {
      "set": 0,
      "father": {
        "birth": "29 September 1969",
        "death": "",
        "displayName": "Wayne Gordon Gault"
      },
      "fatherId": "102281560836",
      "mother": {
        "birth": "15 May 1963",
        "death": "",
        "displayName": "Mary Ann Sutcliffe"
      },
      "motherId": "102281560706",
      "surname": "Gault"
    }
  }
}
```

**Key Field:** `apmFindExistingPerson.PID` - The person ID of the existing person to link.

**Use Cases:**
- Link DNA match to family tree
- Connect already-existing person as parent/child/spouse
- Avoid creating duplicate persons

**Relationship Types:**
| type | Description |
|------|-------------|
| `Spouse` | Spouse/partner |
| `Child` | Child of person |
| `Father` | Father (unverified) |
| `Mother` | Mother (unverified) |
| `Sibling` | Sibling (unverified) |

**Response:** JSON with new person ID and relationship data

---

## 4. Remove Person

Removes a person from the tree entirely.

**Endpoint:**
```
POST /family-tree/person/tree/{treeId}/person/{personId}/removePerson
```

**Headers:**
```
Content-Type: application/x-www-form-urlencoded
X-Requested-With: XMLHttpRequest
```

**Request Body:**
```
name=FirstName%20LastName
```

**Note:** The `name` parameter must match the person's display name (URL-encoded).

---

## 5. Get Relationship Form (Pre-add)

Gets the form data for adding a specific relationship type.

**Endpoint:**
```
GET /family-tree/person/addedit/user/{userId}/tree/{treeId}/person/{personId}/add?rel={relType}
```

**Relationship Codes:**
| rel | Relationship |
|-----|--------------|
| `w` | Wife/Spouse |
| `h` | Husband/Spouse |
| `c` | Child |
| `f` | Father |
| `m` | Mother |
| `s` | Sibling |

---

## 6. Get Existing Relationships

Retrieves all current relationships for a person. Useful for reading relationship state before modifications.

**Endpoint:**
```
GET /family-tree/person/addedit/user/{userId}/tree/{treeId}/person/{personId}/editrelationships
```

**Response:** JSON containing:
- Parents (father, mother, alternate parents)
- Spouses/partners
- Children
- Siblings

**Use Cases:**
- Verify relationship exists before modification
- Get relationship IDs for updates
- Display current family structure

---

## 7. Remove Relationship

Removes a relationship between two people **without deleting either person**. The related person remains in the tree but is no longer connected.

**Endpoint:**
```
POST /family-tree/person/addedit/user/{userId}/tree/{treeId}/person/{personId}/relationship/{relatedPersonId}/removerelationship
```

**Request Body:**
```json
{
  "type": "C",
  "parentType": "M"
}
```

**Type Codes:**
| Code | Meaning |
|------|---------|
| `C` | Child |
| `F` | Father |
| `M` | Mother |
| `H` | Husband/Spouse |
| `W` | Wife/Spouse |
| `S` | Sibling |

**parentType**: Indicates which parent role the `personId` has in this relationship (M = Mother, F = Father)

**Use Cases:**
- Correct mistaken parent-child links
- Remove duplicate relationships
- Unlink merged persons

---

## 8. Change Relationship Type

Modifies the type of an existing relationship (e.g., spouse → ex-spouse, biological → adoptive).

**Endpoint:**
```
POST /family-tree/person/addedit/user/{userId}/tree/{treeId}/person/{personId}/relationship/{relatedPersonId}/changerelationship
```

**Request Body:**
```json
{
  "modifier": "spu",
  "originalModifier": "sps",
  "type": "H",
  "parentType": "M",
  "pty": -1
}
```

**Modifier Codes (Spouse Relationships):**
| Code | Meaning |
|------|---------|
| `sps` | Spouse (current) |
| `spu` | Ex-Spouse (former) |
| `spp` | Partner |

**Modifier Codes (Parent Relationships):**
| Code | Meaning |
|------|---------|
| `bio` | Biological |
| `adp` | Adoptive |
| `fos` | Foster |
| `stp` | Step-parent |
| `gua` | Guardian |

**Note:** The exact modifier codes are inferred from patterns; actual values may vary.

**Use Cases:**
- Mark spouse as ex-spouse after divorce
- Change biological parent to adoptive
- Correct relationship type errors

---

## 9. Add Web Link

Adds an external URL/web link to a person's profile (appears in Facts section).

**Endpoint:**
```
POST /family-tree/person/facts/user/{userId}/tree/{treeId}/person/{personId}/weblinkadd
```

**Request Body:**
```json
{
  "webLinkHref": "https://example.com/article",
  "webLinkTitle": "News article about person"
}
```

**Use Cases:**
- Link to external research sources
- Reference online obituaries
- Connect to newspaper archives

---

## Implementation Notes

### SuggestedFact Mapping

When applying `SuggestedFact` records to the tree:

```python
# FactTypeEnum to API mapping
FACT_TYPE_TO_API = {
    FactTypeEnum.BIRTH: {
        "api": "updatePerson",
        "fields": {"bdate": "date", "bplace": "place"}
    },
    FactTypeEnum.DEATH: {
        "api": "updatePerson",
        "fields": {"ddate": "date", "dplace": "place"}
    },
    FactTypeEnum.MARRIAGE: {
        "api": "factedit",
        "eventType": "marriage"
    },
    FactTypeEnum.LOCATION: {
        "api": "factedit",
        "eventType": "residence"
    },
    FactTypeEnum.RELATIONSHIP: {
        "api": "addperson",
        "type": "dynamic"  # Spouse/Child/Parent based on context
    },
    FactTypeEnum.OTHER: {
        "api": "factedit",
        "eventType": "dynamic"  # Infer from fact content
    }
}
```

### Rate Limiting Considerations

- Use existing `RateLimiter` from `core/rate_limiter.py`
- Recommend: 1 write per 3 seconds minimum
- Ancestry may have stricter limits on write operations

### Error Handling

Expected error responses:
- `401` - Session expired, refresh cookies
- `403` - Insufficient permissions
- `404` - Person/tree not found
- `409` - Conflict (data changed)
- `429` - Rate limited

### Audit Trail

All write operations should be logged to:
1. Application log (`Logs/app.log`)
2. `TreeUpdateLog` table (to be created)
3. `SuggestedFact.applied_at` timestamp

---

## Phase 8 Implementation Checklist

- [x] Create `TreeUpdateService` class in `api/tree_update.py`
- [ ] Implement `apply_suggested_fact()` method
- [x] Add `TreeUpdateLog` model to `database.py`
- [ ] Wire into approval workflow (Action Review)
- [ ] Add post-update verification (re-fetch and compare)
- [ ] Integration tests with mock responses

---

## Additional APIs (To Be Discovered)

| Operation | Status | Notes |
|-----------|--------|-------|
| Add source citation | ❓ Not captured | Attach source to fact |
| Upload media/photo | ❓ Not captured | Different API path expected |

---

## Version History

| Date | Changes |
|------|---------|
| 2025-12-17 | Initial discovery: updatePerson, factedit, addperson, removePerson |
| 2025-12-17 | Added: editrelationships (GET), confirmed `Person` gender for unknown |
| 2025-12-17 | Added: removerelationship, changerelationship (relationship management) |
| 2025-12-17 | Added: Link existing person (apmFindExistingPerson.PID), weblinkadd |
