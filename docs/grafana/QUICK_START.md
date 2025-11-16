# Grafana Quick Start Guide

## Step 1: Install Grafana

### Windows Installation

1. **Download Grafana:**
   - Go to: https://grafana.com/grafana/download?platform=windows
   - Download the **Windows Installer (.msi)** for the latest stable version

2. **Run Installer:**
   - Double-click the `.msi` file
   - Follow installation wizard (use default settings)
   - Default install location: `C:\Program Files\GrafanaLabs\grafana\`

3. **Start Grafana Service:**
   ```powershell
   # Start the service
   Start-Service Grafana
   
   # Verify it's running
   Get-Service Grafana
   
   # Set to start automatically on boot
   Set-Service -Name Grafana -StartupType Automatic
   ```

4. **Access Grafana:**
   - Open browser: http://localhost:3000
   - Default login: `admin` / `admin`
   - You'll be prompted to change password on first login

---

## Step 2: Install SQLite Plugin

**After Grafana is installed:**

```powershell
# Navigate to Grafana bin directory
cd "C:\Program Files\GrafanaLabs\grafana\bin"

# Install SQLite plugin
.\grafana-cli plugins install frser-sqlite-datasource

# Restart Grafana service
Restart-Service Grafana
```

---

## Step 3: Configure Data Sources

### A. Prometheus Data Source (System Metrics)

1. Open Grafana → ⚙️ Configuration → Data Sources
2. Click **Add data source**
3. Select **Prometheus**
4. Configure:
   - **Name:** `Prometheus`
   - **URL:** `http://localhost:9090` ⚠️ **Port 9090** (not 9000)
   - **Access:** `Server (default)`
5. Click **Save & Test** → Should see "✅ Data source is working"

### B. SQLite Data Source (Genealogy Data)

1. Click **Add data source** again
2. Select **SQLite**
3. Configure:
   - **Name:** `SQLite`
   - **Path:** `C:\Users\wayne\GitHub\Python\Projects\Ancestry\Data\ancestry.db`
   - ⚠️ **Use your actual username in path**
4. Click **Save & Test**

---

## Step 4: Import Dashboards

### Dashboard 1: System Performance

1. Click **+** (Create) → **Import**
2. Click **Upload JSON file**
3. Select: `C:\Users\wayne\GitHub\Python\Projects\Ancestry\docs\grafana\system_performance.json`
4. In dropdown: Select **Prometheus** data source
5. Click **Import**

### Dashboard 2: Genealogy Insights

1. Click **+** → **Import** again
2. Upload: `docs\grafana\genealogy_insights.json`
3. Configure:
   - **DS_PROMETHEUS:** Select `Prometheus`
   - **DS_SQLITE:** Select `SQLite`
4. Click **Import**

---

## Step 5: Start Metrics Collection

```powershell
# Run the application
python main.py

# Select option "M" to view metrics
# Both dashboards will open automatically
```

---

## Troubleshooting

### "Grafana service not found"
**Cause:** Grafana not installed yet  
**Fix:** Follow Step 1 above

### "grafana-cli not recognized"
**Cause:** Grafana not in PATH, or not installed  
**Fix:** Use full path: `cd "C:\Program Files\GrafanaLabs\grafana\bin"` then `.\grafana-cli`

### "Can't connect to Prometheus"
**Cause:** Prometheus not running or wrong port  
**Fix:**
1. Start your Python app: `python main.py`
2. Verify Prometheus is running: Open http://localhost:9090
3. Check Grafana data source uses port **9090** (not 9000)

### "SQLite panels show no data"
**Cause:** Database path incorrect or no data  
**Fix:**
1. Run Action 6 to collect DNA matches: `python main.py` → Option "6"
2. Verify database has data:
   ```powershell
   sqlite3 Data\ancestry.db "SELECT COUNT(*) FROM people;"
   ```
3. Check SQLite data source path in Grafana matches your actual path

### Panels show "Data source not found"
**Cause:** Data source UID mismatch  
**Fix:** Edit dashboard JSON → Replace `${DS_PROMETHEUS}` with your data source UID

---

## Quick Verification Checklist

- [ ] Grafana installed and service running
- [ ] SQLite plugin installed (`grafana-cli plugins list` shows it)
- [ ] Prometheus data source configured (port 9090)
- [ ] SQLite data source configured (correct database path)
- [ ] System Performance dashboard imported
- [ ] Genealogy Insights dashboard imported
- [ ] Python app running (`python main.py`)
- [ ] Dashboards accessible at http://localhost:3000/d/ancestry-performance

---

## Alternative: Portable Grafana (No Installation)

If you prefer not to install as a service:

1. **Download Standalone ZIP:**
   - https://grafana.com/grafana/download?platform=windows
   - Select "Standalone Windows Binaries"

2. **Extract and Run:**
   ```powershell
   # Extract to C:\Programs\Grafana
   cd C:\Programs\Grafana\bin
   .\grafana-server.exe
   ```

3. **Continue from Step 2** above (install plugin, configure data sources)

---

**Need Help?** See full documentation: `DASHBOARD_SETUP.md`
