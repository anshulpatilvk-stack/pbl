from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import BaseModel
from database import init_db, SessionLocal, User, SaleRecord, StockRecord, CompetitorPrice
from vision_model import get_detailed_freshness_report

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.on_event("startup")
def startup_event():
    init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserAuth(BaseModel):
    username: str
    password: str

class SaleCreate(BaseModel):
    username: str
    product: str
    quantity_sold: float
    total_revenue: float
    sale_date: str

class StockUpdate(BaseModel):
    username: str
    product: str
    stock_quantity: float

class CompetitorCreate(BaseModel):
    username: str
    product: str
    competitor_name: str
    price_per_unit: float

class PriceCalculationRequest(BaseModel):
    category: str
    freshness_score: float
    competitor_avg: float = 0.0

class RecordDeleteRequest(BaseModel):
    username: str
    record_id: int
    record_type: str  # "sale", "stock", or "competitor"

@app.post("/register")
def register(user_data: UserAuth, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = pwd_context.hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/login")
def login(user_data: UserAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user or not pwd_context.verify(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"message": "Login successful", "username": user.username}

@app.post("/analyze/")
async def analyze_produce(file: UploadFile = File(...), category: str = Form(...)):
    try:
        image_bytes = await file.read()
        report = get_detailed_freshness_report(image_bytes, category=category)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/calculate-price")
def calculate_system_price(data: PriceCalculationRequest):
    base_prices = {
        "Tomato": 40.0,
        "Apple": 120.0,
        "Banana": 50.0,
        "Lemon": 60.0
    }
    
    base_price = base_prices.get(data.category, 50.0)
    freshness_factor = data.freshness_score / 100.0
    freshness_adjusted_price = base_price * freshness_factor
    
    if data.competitor_avg > 0:
        recommended_price = (0.6 * freshness_adjusted_price) + (0.4 * data.competitor_avg)
        if data.freshness_score >= 90:
            recommended_price = max(recommended_price, data.competitor_avg * 0.95)
        else:
            recommended_price = min(recommended_price, data.competitor_avg * 0.90)
    else:
        recommended_price = freshness_adjusted_price
        
    return {
        "category": data.category,
        "base_market_price": base_price,
        "recommended_system_price": round(recommended_price, 2),
        "pricing_strategy": "Optimized for freshness grade and competitor benchmark."
    }

@app.post("/api/sale")
def log_sale(data: SaleCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    sale = SaleRecord(
        user_id=user.id,
        product=data.product,
        quantity_sold=data.quantity_sold,
        total_revenue=data.total_revenue,
        sale_date=data.sale_date
    )
    db.add(sale)
    db.commit()
    return {"message": "Sale logged successfully"}

@app.post("/api/stock")
def update_stock(data: StockUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    stock = db.query(StockRecord).filter(StockRecord.user_id == user.id, StockRecord.product == data.product).first()
    if stock:
        stock.stock_quantity = data.stock_quantity
    else:
        stock = StockRecord(user_id=user.id, product=data.product, stock_quantity=data.stock_quantity)
        db.add(stock)
    db.commit()
    return {"message": "Stock updated successfully"}

@app.post("/api/competitor")
def add_competitor(data: CompetitorCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    comp = CompetitorPrice(
        user_id=user.id,
        product=data.product,
        competitor_name=data.competitor_name,
        price_per_unit=data.price_per_unit
    )
    db.add(comp)
    db.commit()
    return {"message": "Competitor price recorded"}

@app.delete("/api/record")
def delete_record(data: RecordDeleteRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if data.record_type == "sale":
        record = db.query(SaleRecord).filter(SaleRecord.id == data.record_id, SaleRecord.user_id == user.id).first()
    elif data.record_type == "stock":
        record = db.query(StockRecord).filter(StockRecord.id == data.record_id, StockRecord.user_id == user.id).first()
    elif data.record_type == "competitor":
        record = db.query(CompetitorPrice).filter(CompetitorPrice.id == data.record_id, CompetitorPrice.user_id == user.id).first()
    else:
        raise HTTPException(status_code=400, detail="Invalid record type")
        
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
        
    db.delete(record)
    db.commit()
    return {"message": f"{data.record_type.capitalize()} record deleted successfully"}

@app.get("/api/dashboard/{username}")
def get_dashboard(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    sales = db.query(SaleRecord).filter(SaleRecord.user_id == user.id).all()
    stocks = db.query(StockRecord).filter(StockRecord.user_id == user.id).all()
    competitors = db.query(CompetitorPrice).filter(CompetitorPrice.user_id == user.id).all()
    
    products = ["Tomato", "Apple", "Banana", "Lemon"]
    
    user_avg_prices = {}
    for prod in products:
        prod_sales = [s for s in sales if s.product == prod]
        total_rev = sum(s.total_revenue for s in prod_sales)
        total_qty = sum(s.quantity_sold for s in prod_sales)
        user_avg_prices[prod] = round(total_rev / total_qty, 2) if total_qty > 0 else 0.0

    comp_avg_prices = {}
    for prod in products:
        prod_comps = [c for c in competitors if c.product == prod]
        prices = [c.price_per_unit for c in prod_comps]
        comp_avg_prices[prod] = round(sum(prices) / len(prices), 2) if prices else 0.0
    
    stock_list = [{"id": s.id, "product": s.product, "stock_quantity": s.stock_quantity, "unit": s.unit} for s in stocks]
    competitor_list = [{"id": c.id, "product": c.product, "competitor_name": c.competitor_name, "price_per_unit": c.price_per_unit} for c in competitors]
    sale_list = [{"id": s.id, "product": s.product, "quantity_sold": s.quantity_sold, "total_revenue": s.total_revenue, "sale_date": s.sale_date} for s in sales]

    return {
        "user_avg_prices": user_avg_prices,
        "comp_avg_prices": comp_avg_prices,
        "stocks": stock_list,
        "competitors": competitor_list,
        "sales": sale_list
    }