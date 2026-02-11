"""
Browser-based Review Queue Interface.

Provides a web UI for reviewing, approving, rejecting, and rewriting AI-generated drafts.
Run with: python -m ui.review_server
Access at: http://localhost:5000
"""

import json
import logging
import os
import webbrowser
from datetime import UTC, datetime
from threading import Timer
from typing import Any

from flask import Flask, jsonify, render_template_string, request

from ai.ai_interface import generate_genealogical_reply
from core.database import ConversationLog, DraftReply, Person
from core.session_manager import SessionManager

# Setup logging
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# HTML Template with embedded CSS and JS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review Queue - Ancestry Automation</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --accent-yellow: #fbbf24;
            --accent-blue: #60a5fa;
            --accent-purple: #a78bfa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
            font-size: 16px;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; font-size: 1.1rem; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--bg-card);
        }
        h1 { font-size: 1.8rem; display: flex; align-items: center; gap: 10px; }
        .stats {
            display: flex;
            gap: 20px;
        }
        .stat-box {
            background: var(--bg-card);
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 1.5rem; font-weight: bold; }
        .stat-label { font-size: 0.8rem; color: var(--text-secondary); }

        .draft-list { display: flex; flex-direction: column; gap: 20px; }

        .draft-card {
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--bg-card);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .draft-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        .draft-header {
            background: var(--bg-card);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .draft-id { font-weight: bold; font-size: 1.1rem; }
        .draft-meta { display: flex; gap: 15px; align-items: center; }
        .confidence {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .confidence.high { background: var(--accent-green); color: #000; }
        .confidence.medium { background: var(--accent-yellow); color: #000; }
        .confidence.low { background: var(--accent-red); color: #000; }
        .priority {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
        }
        .priority.critical { background: var(--accent-red); color: #000; }
        .priority.high { background: var(--accent-yellow); color: #000; }
        .priority.normal { background: var(--accent-blue); color: #000; }
        .priority.low { background: var(--accent-green); color: #000; }

        .draft-body { padding: 20px; }
        .person-info {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--bg-card);
        }
        .person-name { font-size: 1.4rem; font-weight: 500; }
        .person-details { color: var(--text-secondary); font-size: 1rem; margin-top: 5px; }

        .conversation-section {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            max-height: 300px;
            overflow-y: auto;
        }
        .conversation-section h3 {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .message {
            margin-bottom: 12px;
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 1.05rem;
            line-height: 1.5;
        }
        .message.inbound {
            background: var(--bg-card);
            margin-right: 40px;
        }
        .message.outbound {
            background: var(--accent-purple);
            color: #000;
            margin-left: 40px;
        }
        .message-time {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }
        .message.outbound .message-time { color: rgba(0,0,0,0.6); }

        .draft-content {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid var(--accent-blue);
        }
        .draft-content h3 {
            font-size: 0.9rem;
            color: var(--accent-blue);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .draft-text {
            white-space: pre-wrap;
            line-height: 1.7;
            font-size: 1.1rem;
        }

        .action-bar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1.05rem;
            font-weight: 500;
            transition: opacity 0.2s, transform 0.1s;
        }
        .btn:hover { opacity: 0.9; }
        .btn:active { transform: scale(0.98); }
        .btn-approve { background: var(--accent-green); color: #000; }
        .btn-reject { background: var(--accent-red); color: #000; }
        .btn-rewrite { background: var(--accent-purple); color: #000; }
        .btn-secondary { background: var(--bg-card); color: var(--text-primary); }

        .rewrite-form {
            display: none;
            margin-top: 15px;
            padding: 15px;
            background: var(--bg-primary);
            border-radius: 8px;
        }
        .rewrite-form.active { display: block; }
        .rewrite-form textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--bg-card);
            border-radius: 6px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 0.95rem;
            resize: vertical;
            min-height: 80px;
            margin-bottom: 10px;
        }
        .rewrite-form textarea:focus {
            outline: none;
            border-color: var(--accent-purple);
        }

        .reject-form {
            display: none;
            margin-top: 15px;
            padding: 15px;
            background: var(--bg-primary);
            border-radius: 8px;
        }
        .reject-form.active { display: block; }
        .reject-form input {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--bg-card);
            border-radius: 6px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 0.95rem;
            margin-bottom: 10px;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }
        .empty-state h2 { color: var(--accent-green); margin-bottom: 10px; }

        .loading {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .loading.active { display: flex; }
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid var(--bg-card);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 8px;
            color: #000;
            font-weight: 500;
            transform: translateX(150%);
            transition: transform 0.3s;
            z-index: 1001;
        }
        .toast.show { transform: translateX(0); }
        .toast.success { background: var(--accent-green); }
        .toast.error { background: var(--accent-red); }

        .refresh-btn {
            background: transparent;
            border: 1px solid var(--text-secondary);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .refresh-btn:hover {
            background: var(--bg-card);
            border-color: var(--text-primary);
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📋 Review Queue</h1>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value" id="pending-count">{{ stats.pending_count }}</div>
                    <div class="stat-label">Pending</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="approved-today">{{ stats.approved_today }}</div>
                    <div class="stat-label">Approved Today</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="rejected-today">{{ stats.rejected_today }}</div>
                    <div class="stat-label">Rejected Today</div>
                </div>
                <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
            </div>
        </header>

        <div class="draft-list">
            {% if drafts %}
                {% for draft in drafts %}
                <div class="draft-card" id="draft-{{ draft.id }}">
                    <div class="draft-header">
                        <span class="draft-id">Draft #{{ draft.id }}</span>
                        <div class="draft-meta">
                            <span class="confidence {{ 'high' if draft.confidence >= 85 else 'medium' if draft.confidence >= 70 else 'low' }}">
                                {{ draft.confidence }}% confidence
                            </span>
                            <span class="priority {{ draft.priority }}">{{ draft.priority }}</span>
                            <span style="color: var(--text-secondary); font-size: 0.85rem;">
                                {{ draft.created_at }}
                            </span>
                        </div>
                    </div>
                    <div class="draft-body">
                        <div class="person-info">
                            <div class="person-name">{{ draft.person_name }}</div>
                            <div class="person-details">
                                Person ID: {{ draft.person_id }} |
                                Conversation: {{ draft.conversation_id[:20] }}...
                            </div>
                        </div>

                        {% if draft.conversation %}
                        <div class="conversation-section">
                            <h3>📨 Conversation History</h3>
                            {% for msg in draft.conversation %}
                            <div class="message {{ 'inbound' if msg.direction == 'IN' else 'outbound' }}">
                                <div class="message-time">{{ msg.timestamp }}</div>
                                <div>{{ msg.content }}</div>
                            </div>
                            {% endfor %}
                        </div>
                        {% endif %}

                        <div class="draft-content">
                            <h3>📝 Draft Reply</h3>
                            <div class="draft-text">{{ draft.content }}</div>
                        </div>

                        <div class="action-bar">
                            <button class="btn btn-approve" onclick="approveDraft({{ draft.id }})">
                                ✓ Approve
                            </button>
                            <button class="btn btn-reject" onclick="toggleRejectForm({{ draft.id }})">
                                ✗ Reject
                            </button>
                            <button class="btn btn-rewrite" onclick="toggleRewriteForm({{ draft.id }})">
                                ✎ Rewrite with Feedback
                            </button>
                        </div>

                        <div class="reject-form" id="reject-form-{{ draft.id }}">
                            <input type="text" id="reject-reason-{{ draft.id }}"
                                   placeholder="Reason for rejection (optional)">
                            <button class="btn btn-reject" onclick="rejectDraft({{ draft.id }})">
                                Confirm Rejection
                            </button>
                            <button class="btn btn-secondary" onclick="toggleRejectForm({{ draft.id }})">
                                Cancel
                            </button>
                        </div>

                        <div class="rewrite-form" id="rewrite-form-{{ draft.id }}">
                            <textarea id="rewrite-feedback-{{ draft.id }}"
                                      placeholder="Describe how the AI should rewrite this draft...&#10;&#10;Examples:&#10;• Make it more formal&#10;• Ask about their Smith ancestors specifically&#10;• Shorten it and be more direct"></textarea>
                            <button class="btn btn-rewrite" onclick="rewriteDraft({{ draft.id }})">
                                🔄 Regenerate Draft
                            </button>
                            <button class="btn btn-secondary" onclick="toggleRewriteForm({{ draft.id }})">
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="empty-state">
                    <h2>✅ All caught up!</h2>
                    <p>No pending drafts to review.</p>
                </div>
            {% endif %}
        </div>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        function showLoading() {
            document.getElementById('loading').classList.add('active');
        }

        function hideLoading() {
            document.getElementById('loading').classList.remove('active');
        }

        function showToast(message, type) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast ' + type + ' show';
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        function toggleRewriteForm(id) {
            const form = document.getElementById('rewrite-form-' + id);
            form.classList.toggle('active');
            // Close reject form if open
            document.getElementById('reject-form-' + id).classList.remove('active');
        }

        function toggleRejectForm(id) {
            const form = document.getElementById('reject-form-' + id);
            form.classList.toggle('active');
            // Close rewrite form if open
            document.getElementById('rewrite-form-' + id).classList.remove('active');
        }

        async function approveDraft(id) {
            showLoading();
            try {
                const response = await fetch('/api/approve/' + id, { method: 'POST' });
                const data = await response.json();
                hideLoading();
                if (data.success) {
                    showToast('Draft #' + id + ' approved!', 'success');
                    document.getElementById('draft-' + id).style.display = 'none';
                    updateStats();
                } else {
                    showToast('Error: ' + data.message, 'error');
                }
            } catch (e) {
                hideLoading();
                showToast('Error approving draft', 'error');
            }
        }

        async function rejectDraft(id) {
            const reason = document.getElementById('reject-reason-' + id).value;
            showLoading();
            try {
                const response = await fetch('/api/reject/' + id, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ reason: reason })
                });
                const data = await response.json();
                hideLoading();
                if (data.success) {
                    showToast('Draft #' + id + ' rejected', 'success');
                    document.getElementById('draft-' + id).style.display = 'none';
                    updateStats();
                } else {
                    showToast('Error: ' + data.message, 'error');
                }
            } catch (e) {
                hideLoading();
                showToast('Error rejecting draft', 'error');
            }
        }

        async function rewriteDraft(id) {
            const feedback = document.getElementById('rewrite-feedback-' + id).value;
            if (!feedback.trim()) {
                showToast('Please provide feedback for the rewrite', 'error');
                return;
            }
            showLoading();
            try {
                const response = await fetch('/api/rewrite/' + id, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ feedback: feedback })
                });
                const data = await response.json();
                hideLoading();
                if (data.success) {
                    showToast('Draft #' + id + ' rewritten!', 'success');
                    // Update the draft content on the page
                    const card = document.getElementById('draft-' + id);
                    const draftText = card.querySelector('.draft-text');
                    draftText.textContent = data.new_content;
                    toggleRewriteForm(id);
                    document.getElementById('rewrite-feedback-' + id).value = '';
                } else {
                    showToast('Error: ' + data.message, 'error');
                }
            } catch (e) {
                hideLoading();
                showToast('Error rewriting draft', 'error');
            }
        }

        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                document.getElementById('pending-count').textContent = data.pending_count;
                document.getElementById('approved-today').textContent = data.approved_today;
                document.getElementById('rejected-today').textContent = data.rejected_today;
            } catch (e) {
                console.error('Error updating stats', e);
            }
        }
    </script>
</body>
</html>
"""


def get_db_session() -> Any | None:
    """Get database session from SessionManager."""
    from testing.test_utilities import create_test_database

    try:
        from core.session_manager import SessionManager

        sm = SessionManager()
        return sm.db_manager.get_session()
    except Exception as exc:
        logger.debug("Falling back to in-memory test database for review server: %s", exc)
        try:
            return create_test_database()
        except Exception as inner:  # pragma: no cover - defensive
            logger.error(f"Could not create test database: {inner}")
            return None


def get_queue_stats() -> dict[str, Any]:
    """Get queue statistics."""
    try:
        from core.approval_queue import ApprovalQueueService

        db_session = get_db_session()
        if not db_session:
            return {"pending_count": 0, "approved_today": 0, "rejected_today": 0}

        service = ApprovalQueueService(db_session)
        stats = service.get_queue_stats()
        return {
            "pending_count": stats.pending_count,
            "approved_today": stats.approved_today,
            "rejected_today": stats.rejected_today,
            "auto_approved_count": stats.auto_approved_count,
            "expired_count": stats.expired_count,
        }
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        return {"pending_count": 0, "approved_today": 0, "rejected_today": 0}


def get_pending_drafts(limit: int = 20) -> list[dict[str, Any]]:
    """Get pending drafts with conversation context."""
    try:
        from core.approval_queue import ApprovalQueueService
        from core.database import ConversationLog

        db_session = get_db_session()
        if not db_session:
            return []

        service = ApprovalQueueService(db_session)
        pending = service.get_pending_queue(limit=limit)

        drafts: list[dict[str, Any]] = []
        for draft in pending:
            # Get conversation history
            conv_logs = (
                db_session.query(ConversationLog)
                .filter(ConversationLog.conversation_id == draft.conversation_id)
                .order_by(ConversationLog.latest_timestamp.asc())
                .limit(10)
                .all()
            )

            conversation: list[dict[str, str]] = []
            for log in conv_logs:
                conversation.append(
                    {
                        "direction": log.direction.value,
                        "content": log.latest_message_content or "(no content)",
                        "timestamp": log.latest_timestamp.strftime("%Y-%m-%d %H:%M"),
                    }
                )

            drafts.append(
                {
                    "id": draft.draft_id,
                    "person_id": draft.person_id,
                    "person_name": draft.person_name,
                    "conversation_id": draft.conversation_id,
                    "content": draft.content,
                    "confidence": draft.ai_confidence,
                    "priority": getattr(draft.priority, "value", "normal"),
                    "created_at": draft.created_at.strftime("%Y-%m-%d %H:%M"),
                    "conversation": conversation,
                }
            )

        return drafts
    except Exception as e:
        logger.error(f"Error getting pending drafts: {e}")
        return []


@app.route("/")
def index() -> str:
    """Main review queue page."""
    stats = get_queue_stats()
    drafts = get_pending_drafts()
    return render_template_string(HTML_TEMPLATE, stats=stats, drafts=drafts)


def _json_error(message: str) -> Any:
    """Return a standardized JSON error response."""
    return jsonify({"success": False, "message": message})


@app.route("/api/stats")
def api_stats() -> Any:
    """Get queue statistics as JSON."""
    return jsonify(get_queue_stats())


@app.route("/api/approve/<int:draft_id>", methods=["POST"])
def api_approve(draft_id: int):
    """Approve a draft."""
    try:
        from core.approval_queue import ApprovalQueueService

        db_session = get_db_session()
        if not db_session:
            return jsonify({"success": False, "message": "Database connection failed"})

        service = ApprovalQueueService(db_session)
        result = service.approve(draft_id, reviewer="web_ui")

        return jsonify({"success": result.success, "message": result.message})
    except Exception as e:
        logger.error(f"Error approving draft: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route("/api/reject/<int:draft_id>", methods=["POST"])
def api_reject(draft_id: int):
    """Reject a draft."""
    try:
        from core.approval_queue import ApprovalQueueService

        data: dict[str, Any] = request.get_json() or {}
        reason = str(data.get("reason", ""))

        db_session = get_db_session()
        if not db_session:
            return jsonify({"success": False, "message": "Database connection failed"})

        service = ApprovalQueueService(db_session)
        result = service.reject(draft_id, reviewer="web_ui", reason=reason)

        return jsonify({"success": result.success, "message": result.message})
    except Exception as e:
        logger.error(f"Error rejecting draft: {e}")
        return jsonify({"success": False, "message": str(e)})


def _validate_rewrite_request(
    draft_id: int,
    feedback: str,
    sm: "SessionManager",
) -> tuple[Any, Any, Any, str | None]:
    """Validate rewrite request and return (db_session, draft, person, error_message).

    Returns (None, None, None, error_msg) if validation fails,
    or (db_session, draft, person, None) if validation succeeds.
    """
    if not feedback:
        return None, None, None, "Feedback is required"

    db_session = sm.db_manager.get_session()
    if not db_session:
        return None, None, None, "Database connection failed"

    draft = db_session.query(DraftReply).filter(DraftReply.id == draft_id).first()
    if draft is None:
        return None, None, None, f"Draft {draft_id} not found"
    if draft.status != "PENDING":
        return None, None, None, f"Draft is already {draft.status}"

    person = db_session.query(Person).filter(Person.id == draft.people_id).first()
    if person is None:
        return None, None, None, "Person not found"

    return db_session, draft, person, None


@app.route("/api/rewrite/<int:draft_id>", methods=["POST"])
def api_rewrite(draft_id: int):
    """Rewrite a draft with AI using feedback."""
    try:
        json_data: dict[str, Any] = request.get_json() or {}
        feedback = str(json_data.get("feedback", "")).strip()
        sm = SessionManager()

        db_session, draft, person, error = _validate_rewrite_request(draft_id, feedback, sm)
        if error:
            return _json_error(error)

        result = _generate_rewrite(sm, db_session, draft, person, feedback)
        if isinstance(result, str):
            return _json_error(result)
        return result

    except Exception as e:
        logger.error(f"Error rewriting draft: {e}", exc_info=True)
        return _json_error(str(e))


def _generate_rewrite(
    sm: "SessionManager",
    db_session: Any,
    draft: "DraftReply",
    person: "Person",
    feedback: str,
) -> Any:
    """Generate rewritten draft and commit, returning JSON response or error message."""
    person_name = getattr(person, "display_name", None) or getattr(person, "username", None) or "Unknown"

    conv_logs = (
        db_session.query(ConversationLog)
        .filter(ConversationLog.conversation_id == draft.conversation_id)
        .order_by(ConversationLog.latest_timestamp.desc())
        .limit(5)
        .all()
    )

    conversation_context = ""
    user_last_message = ""
    for log in reversed(conv_logs):
        conversation_context += (
            f"[{'THEM' if log.direction.value == 'IN' else 'ME'}]: {log.latest_message_content or ''}\n"
        )
        if log.direction.value == "IN" and not user_last_message:
            user_last_message = log.latest_message_content or ""

    if not user_last_message:
        user_last_message = "(No inbound message found)"

    rewrite_context = (
        f"{conversation_context}\n\n"
        f"[REWRITE INSTRUCTIONS]: The previous draft was rejected. "
        f"User feedback: {feedback}\n"
        f"Previous draft that needs improvement:\n{draft.content}"
    )

    new_reply = generate_genealogical_reply(
        conversation_context=rewrite_context,
        user_last_message=user_last_message,
        genealogical_data_str=json.dumps(
            {
                "person_name": person_name,
                "relationship": getattr(person, "relationship", None),
                "shared_dna": getattr(person, "total_cm", None),
            }
        ),
        session_manager=sm,
    )

    if not new_reply:
        return "AI failed to generate reply"

    draft.content = new_reply
    draft.created_at = datetime.now(UTC)
    db_session.commit()

    return jsonify({"success": True, "message": "Draft rewritten successfully", "new_content": new_reply})


def open_browser() -> None:
    """Open browser after short delay to allow server to start."""
    webbrowser.open("http://localhost:5000")


def run_server(port: int = 5000, open_browser_on_start: bool = True):
    """Run the review queue web server."""
    print("\n" + "=" * 60)
    print("📋 Review Queue Web Interface")
    print("=" * 60)
    print(f"\n🌐 Starting server at http://localhost:{port}")
    print("   Press Ctrl+C to stop\n")

    should_open_browser = open_browser_on_start and not os.getenv("CI") and not os.getenv("PYTEST_CURRENT_TEST")
    if should_open_browser:
        Timer(1.5, open_browser).start()

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


# === TESTS ===


def _test_get_queue_stats_returns_dict() -> bool:
    """Test that get_queue_stats returns a dict with expected keys."""
    stats = get_queue_stats()
    assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"
    assert "pending_count" in stats, "Missing pending_count key"
    assert "approved_today" in stats, "Missing approved_today key"
    return True


def _test_html_template_valid() -> bool:
    """Test that HTML template is valid string."""
    assert isinstance(HTML_TEMPLATE, str), "Template should be string"
    assert "<!DOCTYPE html>" in HTML_TEMPLATE, "Should have DOCTYPE"
    assert "Review Queue" in HTML_TEMPLATE, "Should have title"
    return True


def run_comprehensive_tests() -> bool:
    """Run all module tests."""
    from testing.test_framework import TestSuite, create_standard_test_runner

    def module_tests() -> bool:
        suite = TestSuite("Review Server", "ui/review_server.py")
        suite.start_suite()
        suite.run_test("Queue stats returns dict", _test_get_queue_stats_returns_dict)
        suite.run_test("HTML template valid", _test_html_template_valid)
        return suite.finish_suite()

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    import sys

    # When running as a module (python -m ui.review_server), run tests if env is set
    if os.getenv("RUN_INTERNAL_TESTS") == "1" or (len(sys.argv) > 1 and sys.argv[1] == "--test"):
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        run_server()
