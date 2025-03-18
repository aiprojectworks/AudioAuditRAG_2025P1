from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default='user')


engine = create_engine("mysql+pymysql://audioaudit_user:audioauditrag@localhost/audio_audit_rag")  # MySQL database
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def seed_users():
    session = Session()
    # Check if users already exist
    if session.query(User).count() == 0:
        users = [
            User(username="admin", password="admin123", role="admin"),
            User(username="user1", password="password1", role="user"),
        ]
        session.add_all(users)
        session.commit()
    session.close()
