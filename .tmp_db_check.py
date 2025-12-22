import sqlite3
from pathlib import Path

DB = Path('Data/ancestry.db')
conn = sqlite3.connect(DB)
c = conn.cursor()
queries = {
    'sentiment_non_null': "SELECT COUNT(*) FROM conversation_log c JOIN people p ON c.people_id = p.id WHERE p.deleted_at IS NULL AND c.ai_sentiment IS NOT NULL",
    'sentiment_dist': "SELECT c.ai_sentiment, COUNT(*) FROM conversation_log c JOIN people p ON c.people_id = p.id WHERE p.deleted_at IS NULL AND c.ai_sentiment IS NOT NULL GROUP BY c.ai_sentiment",
    'dna_cm_non_null': "SELECT COUNT(*) FROM dna_match d JOIN people p ON d.people_id=p.id WHERE p.deleted_at IS NULL AND d.cm_dna IS NOT NULL",
    'dna_cm_bins': "SELECT CASE WHEN d.cm_dna >= 400 THEN '400+ cM (Close)' WHEN d.cm_dna >= 90 THEN '90-400 cM (Extended)' WHEN d.cm_dna >= 20 THEN '20-90 cM (Distant)' ELSE '<20 cM (Very Distant)' END as range, COUNT(*) FROM dna_match d JOIN people p ON d.people_id=p.id WHERE p.deleted_at IS NULL GROUP BY range ORDER BY MIN(d.cm_dna) DESC",
}
for name, sql in queries.items():
    try:
        rows = c.execute(sql).fetchall()
        print(name, rows)
    except Exception as e:
        print(name, 'ERROR', e)
conn.close()
