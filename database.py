from sqlalchemy import Column, Integer, String, Float, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "sqlite:///./freshness_pricing.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    sales = relationship("SaleRecord", back_populates="user")
    stocks = relationship("StockRecord", back_populates="user")
    competitors = relationship("CompetitorPrice", back_populates="user")

class SaleRecord(Base):
    __tablename__ = "sales_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product = Column(String)
    quantity_sold = Column(Float)
    total_revenue = Column(Float)
    sale_date = Column(String)
    
    user = relationship("User", back_populates="sales")

class StockRecord(Base):
    __tablename__ = "stock_records"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product = Column(String)
    stock_quantity = Column(Float)
    unit = Column(String, default="kg")
    
    user = relationship("User", back_populates="stocks")

class CompetitorPrice(Base):
    __tablename__ = "competitor_prices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product = Column(String)
    competitor_name = Column(String)
    price_per_unit = Column(Float)
    
    user = relationship("User", back_populates="competitors")

def init_db():
    Base.metadata.create_all(bind=engine)