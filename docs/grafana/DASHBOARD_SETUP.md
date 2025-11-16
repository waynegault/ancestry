# Grafana Dashboard Setup Guide

This project includes **two comprehensive Grafana dashboards** for monitoring system performance and genealogy research insights.

---

## 📊 Dashboard Overview

### 1. **System Performance & Health** (`system_performance.json`)
**UID:** `ancestry-performance`

Monitors technical efficiency, API performance, and system reliability:

#### System Health Overview
- **Session Active Time** - How long the current session has been running
- **Total API Calls Made** - Cumulative API requests across all endpoints
- **Active Worker Threads** - Current parallel processing capacity
- **Session Refresh Count** - How many times session cookies were refreshed

#### API Performance
- **API Response Time by Endpoint (95th Percentile)** - Latency tracking per endpoint
- **API Request Rate (Requests/Second)** - Real-time throughput monitoring
- **API Success Rate vs Errors** - Stacked area chart showing success/error distribution
- **Average Response Time by Endpoint** - Mean latency across all endpoints

#### Rate Limiting & Efficiency
- **Rate Limiter Wait Time (Percentiles)** - p50/p95/p99 wait times (monitors 429 prevention)
- **Actions Processed Over Time** - Track which actions are being executed

---

### 2. **Genealogy Research Insights** (`genealogy_insights.json`)
**UID:** `ancestry-genealogy`

Tracks DNA match collection, communication success, and research progress:

#### DNA Match Collection
- **Total DNA Matches in Database** - Current match count
- **Total People in Database** - Unique individuals tracked
- **Average Shared DNA (cM)** - Mean centimorgans across matches
- **People Linked to Tree** - Matches integrated into your family tree
- **DNA Matches by Relationship** - Pie chart showing distribution (1st cousin, 2nd-3rd, etc.)
- **Shared DNA Distribution (cM)** - Table with ranges (0-20, 20-90, 90-200, etc.)

#### Communication & Engagement
- **Total Conversations** - Unique message threads
- **Messages Received (Inbound)** - Responses from matches
- **Messages Sent (Outbound)** - Your outreach messages
- **Response Rate** - Percentage of matches who replied
- **Message Sentiment Distribution** - AI classification (PRODUCTIVE, DESIST, etc.)
- **Recent Conversations Timeline** - Last 20 messages with sentiment

#### Research Progress & Data Quality
- **Contactable vs Non-Contactable** - Message availability status
- **Person Status Distribution** - ACTIVE, DESIST, ARCHIVE breakdown
- **Data Completeness Score** - Percentage of optional fields filled
- **Top DNA Matches (Closest Relatives)** - 20 highest cM matches with tree status
- **Active vs Inactive Matches** - Last login activity tracking

---

## 🛠️ Setup Instructions

### Prerequisites

1. **Prometheus Server** (data collection engine)
   - Download: https://prometheus.io/download/
   - Extract to: `C:/Programs/Prometheus/`
   - **Automated startup** via `metrics_exporter.py` ✅

2. **Grafana** (visualization platform)
   - Download: https://grafana.com/grafana/download
   - Install and start service
   - Default: http://localhost:3000 (admin/admin)

3. **SQLite Datasource Plugin** (for genealogy data)
   ```bash
   grafana-cli plugins install frser-sqlite-datasource
   # Restart Grafana after installation
   ```

---

### Step-by-Step Configuration

#### 1. Enable Metrics Collection
Add to `.env`:
```env
PROMETHEUS_METRICS_ENABLED=true
PROMETHEUS_METRICS_PORT=9000
```

#### 2. Start Application
```powershell
python main.py
# Prometheus server auto-starts at http://localhost:9090
# Metrics exporter runs at http://localhost:9000/metrics
```

#### 3. Configure Grafana Data Sources

##### **Prometheus Data Source** (for system metrics)
1. Open Grafana → Configuration → Data Sources
2. Click "Add data source" → Select "Prometheus"
3. Configure:
   - **Name:** `Prometheus`
   - **URL:** `http://localhost:9090` ⚠️ **Not 9000** (that's the exporter)
   - **Access:** `Server (default)`
4. Click "Save & Test" → Should see "Data source is working"

##### **SQLite Data Source** (for genealogy data)
1. In Grafana → Configuration → Data Sources
2. Click "Add data source" → Select "SQLite"
3. Configure:
   - **Name:** `SQLite`
   - **Path:** `C:\Users\wayne\GitHub\Python\Projects\Ancestry\Data\ancestry.db`
4. Click "Save & Test"

#### 4. Import Dashboards

##### **System Performance Dashboard**
1. Click "+" → Import dashboard
2. Click "Upload JSON file"
3. Select: `docs/grafana/system_performance.json`
4. Configure:
   - **Prometheus datasource:** Select "Prometheus" (created above)
5. Click "Import"

##### **Genealogy Insights Dashboard**
1. Click "+" → Import dashboard
2. Upload: `docs/grafana/genealogy_insights.json`
3. Configure:
   - **Prometheus datasource:** Select "Prometheus"
   - **SQLite datasource:** Select "SQLite"
4. Click "Import"

---

## 🚀 Usage

### From Main Menu
```powershell
python main.py
# Select: "M" (View Metrics Dashboard)
# Both dashboards open automatically in browser tabs
```

### Manual Access
- **System Performance:** http://localhost:3000/d/ancestry-performance
- **Genealogy Insights:** http://localhost:3000/d/ancestry-genealogy

---

## 🔧 Troubleshooting

### "Dashboard not found"
- **Cause:** Dashboards not imported yet
- **Fix:** Follow "Import Dashboards" section above

### "Data source not found" (red panels)
- **Cause:** Missing Prometheus or SQLite data source configuration
- **Fix:** Configure both data sources (Step 3 above)

### "No data" in panels
- **System Performance:** Run an action (e.g., Action 6) to generate metrics
- **Genealogy Insights:** Ensure database has data (`SELECT COUNT(*) FROM people;`)

### Prometheus connection refused
- **Cause:** Prometheus server not running
- **Fix:** Restart `python main.py` (auto-starts Prometheus)
- **Verify:** Open http://localhost:9090 (should show Prometheus UI)

### SQLite panels showing errors
- **Cause:** Incorrect database path or missing plugin
- **Fix:**
  1. Verify path in data source: `C:\Users\wayne\...\Data\ancestry.db`
  2. Install plugin: `grafana-cli plugins install frser-sqlite-datasource`
  3. Restart Grafana service

---

## 📈 Dashboard Customization

### Adding Variables
Edit JSON files to add filters:
```json
"templating": {
  "list": [
    {
      "name": "time_range",
      "type": "interval",
      "options": ["5m", "15m", "1h", "6h", "24h"]
    }
  ]
}
```

### Creating Alerts
1. Edit panel → Alert tab
2. Configure conditions (e.g., "API errors > 10")
3. Set notification channels (email, Slack, etc.)

### Exporting Snapshots
- Dashboard menu → Share → Snapshot
- Create public/private link for sharing

---

## 📝 Metrics Reference

### Prometheus Metrics (System Performance)
| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `ancestry_api_latency_seconds` | Histogram | endpoint, status_family | Response time distribution |
| `ancestry_api_requests_total` | Counter | endpoint, method, result | Total API calls (success/error) |
| `ancestry_session_uptime_seconds` | Gauge | - | Current session age |
| `ancestry_session_refresh_total` | Counter | reason | Session refresh count |
| `ancestry_worker_thread_count` | Gauge | - | Active parallel workers |
| `ancestry_rate_limiter_delay_seconds` | Histogram | - | Wait time distribution |
| `ancestry_action_processed_total` | Counter | action | Actions executed |

### SQLite Tables (Genealogy Insights)
| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `people` | uuid, username, status, contactable | Match profiles |
| `dna_match` | cm_dna, predicted_relationship, people_id | DNA data |
| `conversation_log` | direction, ai_sentiment, conversation_id | Messages |
| `family_tree` | actual_relationship, in_my_tree, people_id | Tree links |

---

## 🎯 Best Practices

1. **Refresh Rate:** Set to 10-30s for live monitoring, 5m for historical analysis
2. **Time Range:** Use "Last 1 hour" for debugging, "Last 30 days" for trends
3. **Annotations:** Add markers for code deployments, configuration changes
4. **Variables:** Create filters for specific endpoints, match ranges, sentiment types
5. **Alerting:** Set up alerts for:
   - API error rate > 5%
   - Session refresh count > 10
   - Response rate < 10%

---

## 📚 Additional Resources

- **Prometheus Documentation:** https://prometheus.io/docs/
- **Grafana Dashboard Guide:** https://grafana.com/docs/grafana/latest/dashboards/
- **PromQL Query Language:** https://prometheus.io/docs/prometheus/latest/querying/basics/
- **SQLite Datasource Plugin:** https://grafana.com/grafana/plugins/frser-sqlite-datasource/

---

**Created:** 2025-01-24  
**Dashboards:** `system_performance.json`, `genealogy_insights.json`  
**Maintenance:** Update dashboard UIDs if renaming/reorganizing
