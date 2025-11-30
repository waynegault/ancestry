from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, Person, SharedMatch
from genealogy.research_service import ResearchService
from genealogy.triangulation import TriangulationService


def test_triangulation_service():
    # Setup in-memory DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create test data
    p1 = Person(username="Target", uuid="TARGET-UUID")
    p2 = Person(username="Shared1", uuid="SHARED1-UUID")
    p3 = Person(username="Shared2", uuid="SHARED2-UUID")
    session.add_all([p1, p2, p3])
    session.commit()

    # Create shared matches
    sm1 = SharedMatch(person_id=p1.id, shared_match_id=p2.id, shared_cm=50)
    sm2 = SharedMatch(person_id=p1.id, shared_match_id=p3.id, shared_cm=20)
    session.add_all([sm1, sm2])
    session.commit()

    # Mock ResearchService
    mock_research = MagicMock(spec=ResearchService)

    # Initialize Service
    service = TriangulationService(session, mock_research)

    # Test _get_shared_matches
    shared = service._get_shared_matches(p1)
    assert len(shared) == 2
    assert any(p.username == "Shared1" for p in shared)
    assert any(p.username == "Shared2" for p in shared)

    # Test find_triangulation_opportunities
    # Only Shared1 has a path (mocked)
    mock_research.get_relationship_path.side_effect = lambda _root, uuid: ["Path"] if uuid == "SHARED1-UUID" else None

    opportunities = service.find_triangulation_opportunities("TARGET-UUID")
    assert len(opportunities) == 1
    assert opportunities[0]["shared_match"].username == "Shared1"

    print("TriangulationService tests passed!")
    return True


if __name__ == "__main__":
    test_triangulation_service()
