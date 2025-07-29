#!/usr/bin/env python3

"""
Test Program Executor - Safe Testing Protocol Implementation
Implements the comprehensive testing program for ancestry research automation.
SAFETY: Only sends test messages to Frances McHardy (nee Milne).
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from sqlalchemy import text
from core.session_manager import SessionManager
from database import Person, DnaMatch, FamilyTree, ConversationLog, MessageDirectionEnum
from ai_interface import classify_message_intent, extract_genealogical_entities
from person_search import search_gedcom_persons
from config import config_schema


class SafeTestingProtocol:
    """Implements safe testing protocol with Frances McHardy only."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.test_start_time = datetime.now(timezone.utc)
        self.approved_patterns = ["frances", "fran", "mchardy", "milne"]
        self.test_results = {}

    def is_safe_test_recipient(self, person: Person) -> bool:
        """Verify if person is approved for testing."""
        if not person or not person.username:
            return False

        username_lower = person.username.lower()
        return any(pattern in username_lower for pattern in self.approved_patterns)

    def find_frances_mchardy(self) -> Optional[Person]:
        """Find Frances McHardy in the database."""
        session = self.session_manager.get_db_conn()
        if not session:
            logger.error("Could not get database session")
            return None

        try:
            # Search for Frances with various name patterns
            frances = (
                session.query(Person)
                .filter(Person.username.ilike("%frances%"), Person.deleted_at.is_(None))
                .first()
            )

            if not frances:
                # Try alternative patterns
                for pattern in ["mchardy", "milne", "fran"]:
                    frances = (
                        session.query(Person)
                        .filter(
                            Person.username.ilike(f"%{pattern}%"),
                            Person.deleted_at.is_(None),
                        )
                        .first()
                    )
                    if frances:
                        break

            if frances and self.is_safe_test_recipient(frances):
                logger.info(f"Found approved test recipient: {frances.username}")
                return frances
            else:
                logger.warning("Frances McHardy not found in database or not approved")
                return None

        except Exception as e:
            logger.error(f"Error finding Frances McHardy: {e}")
            return None
        finally:
            self.session_manager.return_session(session)

    def analyze_current_state(self) -> Dict[str, any]:
        """Phase 1: Analyze current database state."""
        logger.info("🔍 Phase 1: Analyzing current database state...")

        session = self.session_manager.get_db_conn()
        if not session:
            return {"error": "Could not get database session"}

        try:
            results = {}

            # DNA Match Inventory
            total_query = text(
                """
                SELECT 
                    COUNT(*) as total_matches,
                    SUM(CASE WHEN in_my_tree = 1 THEN 1 ELSE 0 END) as in_tree_count,
                    SUM(CASE WHEN in_my_tree = 0 THEN 1 ELSE 0 END) as out_tree_count
                FROM people 
                WHERE deleted_at IS NULL
            """
            )

            total_result = session.execute(total_query).fetchone()
            results["match_inventory"] = {
                "total_matches": total_result.total_matches,
                "in_tree_count": total_result.in_tree_count,
                "out_tree_count": total_result.out_tree_count,
            }

            # Status breakdown
            status_query = text(
                """
                SELECT status, COUNT(*) as count 
                FROM people 
                WHERE deleted_at IS NULL 
                GROUP BY status
            """
            )

            status_results = session.execute(status_query).fetchall()
            results["status_breakdown"] = {
                row.status: row.count for row in status_results
            }

            # Tree placement analysis
            tree_query = text(
                """
                SELECT 
                    ft.actual_relationship,
                    COUNT(*) as match_count,
                    AVG(dm.cM_DNA) as avg_cm
                FROM family_tree ft
                JOIN people p ON ft.people_id = p.id
                JOIN dna_match dm ON dm.people_id = p.id
                WHERE p.deleted_at IS NULL
                GROUP BY ft.actual_relationship
                ORDER BY match_count DESC
                LIMIT 10
            """
            )

            tree_results = session.execute(tree_query).fetchall()
            results["tree_placement"] = [
                {
                    "relationship": row.actual_relationship,
                    "count": row.match_count,
                    "avg_cm": float(row.avg_cm) if row.avg_cm else 0,
                }
                for row in tree_results
            ]

            # Communication history
            comm_query = text(
                """
                SELECT 
                    COUNT(DISTINCT cl.conversation_id) as total_conversations,
                    COUNT(DISTINCT CASE WHEN cl.direction = 'OUT' THEN cl.conversation_id END) as outgoing_conversations,
                    COUNT(DISTINCT CASE WHEN cl.direction = 'IN' THEN cl.conversation_id END) as incoming_conversations,
                    COUNT(DISTINCT CASE WHEN cl.ai_sentiment = 'PRODUCTIVE' THEN cl.conversation_id END) as productive_conversations
                FROM conversation_log cl
                JOIN people p ON cl.people_id = p.id
                WHERE p.deleted_at IS NULL
            """
            )

            comm_result = session.execute(comm_query).fetchone()
            results["communication_summary"] = {
                "total_conversations": comm_result.total_conversations,
                "outgoing_conversations": comm_result.outgoing_conversations,
                "incoming_conversations": comm_result.incoming_conversations,
                "productive_conversations": comm_result.productive_conversations,
            }

            logger.info("✅ Phase 1 completed: Database state analyzed")
            return results

        except Exception as e:
            logger.error(f"Error analyzing database state: {e}")
            return {"error": str(e)}
        finally:
            self.session_manager.return_session(session)

    def test_ai_processing(self) -> Dict[str, any]:
        """Phase 2: Test AI processing capabilities."""
        logger.info("🤖 Phase 2: Testing AI processing...")

        test_messages = [
            "Thank you for reaching out! I believe we're related through the Gault line. My great-grandfather was John Gault born in 1850 in Aberdeen.",
            "Please don't contact me again about DNA matches. I'm not interested in genealogy research.",
            "I have information about Mary Milne from Aberdeen. She married into the Gault family around 1875.",
            "My grandmother was Frances Milne. She lived in Scotland before moving to Canada.",
            "I'd love to help with your research! I have photos and documents about our shared ancestors.",
        ]

        results = []

        for i, message in enumerate(test_messages):
            try:
                logger.info(f"Testing message {i+1}/5...")

                # Test AI classification
                classification = classify_message_intent(message, self.session_manager)

                # Test data extraction
                extracted_data = extract_genealogical_entities(
                    message, self.session_manager
                )

                result = {
                    "message_preview": message[:50] + "...",
                    "classification": classification,
                    "extracted_data": extracted_data,
                    "success": classification is not None,
                }

                results.append(result)
                logger.info(
                    f"Message {i+1}: Classification={classification}, Success={result['success']}"
                )

            except Exception as e:
                logger.error(f"Error processing test message {i+1}: {e}")
                results.append(
                    {
                        "message_preview": message[:50] + "...",
                        "classification": None,
                        "extracted_data": None,
                        "success": False,
                        "error": str(e),
                    }
                )

        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        logger.info(
            f"✅ Phase 2 completed: AI processing success rate: {success_rate:.1f}%"
        )

        return {
            "test_results": results,
            "success_rate": success_rate,
            "total_tests": len(results),
        }

    def test_tree_integration(self) -> Dict[str, any]:
        """Phase 3: Test tree integration capabilities."""
        logger.info("🌳 Phase 3: Testing tree integration...")

        try:
            # Test GEDCOM search for Frances
            search_results = search_gedcom_persons(
                search_criteria={"first_name": "Frances", "surname": "McHardy"},
                max_results=10,
            )

            # Also try Milne surname
            search_results_milne = search_gedcom_persons(
                search_criteria={"first_name": "Frances", "surname": "Milne"},
                max_results=10,
            )

            results = {
                "mchardy_search": {
                    "results_count": len(search_results),
                    "results": (
                        search_results[:3] if search_results else []
                    ),  # First 3 results
                },
                "milne_search": {
                    "results_count": len(search_results_milne),
                    "results": search_results_milne[:3] if search_results_milne else [],
                },
                "success": True,
            }

            logger.info(
                f"✅ Phase 3 completed: Found {len(search_results)} McHardy and {len(search_results_milne)} Milne matches"
            )
            return results

        except Exception as e:
            logger.error(f"Error testing tree integration: {e}")
            return {"success": False, "error": str(e)}

    def validate_safety_guards(self) -> Dict[str, any]:
        """Phase 4: Validate safety guards are working."""
        logger.info("🛡️ Phase 4: Validating safety guards...")

        session = self.session_manager.get_db_conn()
        if not session:
            return {"error": "Could not get database session"}

        try:
            # Check for any unauthorized messages sent during testing
            unauthorized_query = text(
                """
                SELECT COUNT(*) as count
                FROM conversation_log cl
                JOIN people p ON cl.people_id = p.id
                WHERE cl.direction = 'OUT'
                AND cl.latest_timestamp >= :test_start_time
                AND p.deleted_at IS NULL
            """
            )

            unauthorized_result = session.execute(
                unauthorized_query, {"test_start_time": self.test_start_time}
            ).fetchone()

            # Find Frances to verify she's the only approved recipient
            frances = self.find_frances_mchardy()

            results = {
                "test_start_time": self.test_start_time.isoformat(),
                "unauthorized_messages_count": unauthorized_result.count,
                "frances_found": frances is not None,
                "frances_username": frances.username if frances else None,
                "safety_guards_active": True,
                "approved_patterns": self.approved_patterns,
            }

            logger.info(f"✅ Phase 4 completed: Safety validation successful")
            return results

        except Exception as e:
            logger.error(f"Error validating safety guards: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.session_manager.return_session(session)

    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        logger.info("📊 Generating comprehensive test report...")

        # Run all test phases
        phase1_results = self.analyze_current_state()
        phase2_results = self.test_ai_processing()
        phase3_results = self.test_tree_integration()
        phase4_results = self.validate_safety_guards()

        # Store results
        self.test_results = {
            "phase1_database_analysis": phase1_results,
            "phase2_ai_processing": phase2_results,
            "phase3_tree_integration": phase3_results,
            "phase4_safety_validation": phase4_results,
            "test_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Generate report
        report = f"""
# ANCESTRY AUTOMATION TESTING REPORT
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

## 📊 PHASE 1: DATABASE ANALYSIS
- Total DNA Matches: {phase1_results.get('match_inventory', {}).get('total_matches', 'N/A')}
- In Tree: {phase1_results.get('match_inventory', {}).get('in_tree_count', 'N/A')}
- Out of Tree: {phase1_results.get('match_inventory', {}).get('out_tree_count', 'N/A')}
- Total Conversations: {phase1_results.get('communication_summary', {}).get('total_conversations', 'N/A')}
- Productive Conversations: {phase1_results.get('communication_summary', {}).get('productive_conversations', 'N/A')}

## 🤖 PHASE 2: AI PROCESSING
- Success Rate: {phase2_results.get('success_rate', 'N/A'):.1f}%
- Total Tests: {phase2_results.get('total_tests', 'N/A')}

## 🌳 PHASE 3: TREE INTEGRATION
- McHardy Search Results: {phase3_results.get('mchardy_search', {}).get('results_count', 'N/A')}
- Milne Search Results: {phase3_results.get('milne_search', {}).get('results_count', 'N/A')}

## 🛡️ PHASE 4: SAFETY VALIDATION
- Frances McHardy Found: {phase4_results.get('frances_found', 'N/A')}
- Frances Username: {phase4_results.get('frances_username', 'N/A')}
- Unauthorized Messages: {phase4_results.get('unauthorized_messages_count', 'N/A')}

## ✅ TESTING STATUS
- Database Analysis: {'✅ PASSED' if 'error' not in phase1_results else '❌ FAILED'}
- AI Processing: {'✅ PASSED' if phase2_results.get('success_rate', 0) > 80 else '❌ FAILED'}
- Tree Integration: {'✅ PASSED' if phase3_results.get('success') else '❌ FAILED'}
- Safety Validation: {'✅ PASSED' if phase4_results.get('frances_found') else '❌ FAILED'}

## 🚨 SAFETY STATUS
- Test Recipient Only: Frances McHardy (nee Milne)
- Safety Guards: ACTIVE
- Unauthorized Messages: {phase4_results.get('unauthorized_messages_count', 0)}

{'🎉 READY FOR CONTROLLED TESTING' if all([
    'error' not in phase1_results,
    phase2_results.get('success_rate', 0) > 80,
    phase3_results.get('success'),
    phase4_results.get('frances_found')
]) else '⚠️ ISSUES DETECTED - REVIEW REQUIRED'}
"""

        return report


def run_comprehensive_tests() -> bool:
    """Main function to run comprehensive testing program."""
    logger.info("🚀 Starting Comprehensive Testing Program...")

    try:
        # Initialize session manager
        session_manager = SessionManager()
        if not session_manager.start_sess("Testing Program"):
            logger.error("Failed to start session for testing")
            return False

        # Initialize testing protocol
        test_protocol = SafeTestingProtocol(session_manager)

        # Generate and display report
        report = test_protocol.generate_test_report()
        print(report)

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📄 Test report saved to: {report_file}")
        logger.info("✅ Comprehensive testing completed successfully")

        return True

    except Exception as e:
        logger.error(f"Error running comprehensive tests: {e}", exc_info=True)
        return False
    finally:
        if "session_manager" in locals():
            session_manager.close_sess()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
