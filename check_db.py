from core.database import Person, SessionLocal


def check_db():
    session = SessionLocal()
    try:
        total = session.query(Person).count()
        productive = session.query(Person).filter(Person.sentiment == "PRODUCTIVE").count()
        new_status = session.query(Person).filter(Person.status == "new").count()

        print(f"Total People: {total}")
        print(f"Productive: {productive}")
        print(f"New Status: {new_status}")
    finally:
        session.close()


if __name__ == "__main__":
    check_db()
