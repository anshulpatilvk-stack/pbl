import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Freshness & Inventory Pricing", layout="wide")
API_URL = "http://127.0.0.1:8000"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🔐 Login / Register - Smart Pricing Portal")
    
    with st.form("auth_form"):
        choice = st.selectbox("Action", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            endpoint = "/register" if choice == "Register" else "/login"
            try:
                res = requests.post(f"{API_URL}{endpoint}", json={"username": username, "password": password})
                if res.status_code == 200:
                    if choice == "Register":
                        st.success("Account created successfully! Please switch to Login.")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                else:
                    st.error(res.json().get("detail", "Authentication failed"))
            except Exception as e:
                st.error(f"Could not connect to FastAPI backend server: {e}")

else:
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    tab1, tab2 = st.tabs(["🌱 Produce Freshness & Pricing", "📊 Inventory & Competitor Analytics"])

    with tab1:
        st.header("AI-Driven Freshness Scanner")
        product_category = st.selectbox("Select Product", ["Tomato", "Apple", "Banana", "Lemon"], key="scan_cat")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption=f"Selected: {product_category}", use_container_width=True)
            
            if st.button("Analyze Produce"):
                with st.spinner("Analyzing freshness, shelf life, and characteristics..."):
                    try:
                        files = {"file": uploaded_file.getvalue()}
                        data = {"category": product_category}
                        response = requests.post(f"{API_URL}/analyze/", files=files, data=data)
                        
                        if response.status_code == 200:
                            res_data = response.json()
                            score = res_data.get("freshness_score_100", 0)
                            shelf_life = res_data.get("shelf_life", "N/A")
                            
                            st.session_state.last_score = score
                            st.session_state.last_category = product_category
                            
                            st.success("Analysis Complete!")
                            
                            m1, m2 = st.columns(2)
                            m1.metric("Freshness Score", f"{score}%")
                            m2.metric("Estimated Shelf Life", shelf_life)
                            
                            st.write(f"**Quality Classification:** {res_data.get('classification')}")
                        else:
                            st.error(f"Server Error: {response.text}")
                    except Exception as e:
                        st.error(f"Could not connect to FastAPI backend: {e}")

        # --- DYNAMIC PRICING CALCULATOR SECTION ---
        if "last_score" in st.session_state:
            st.divider()
            st.subheader("💰 Dynamic System Pricing Calculator")
            
            comp_input = st.number_input("Enter Competitor Average Price (₹/kg) [Optional]", min_value=0.0, value=40.0, step=1.0)
            
            if st.button("Calculate System Price"):
                try:
                    pricing_payload = {
                        "category": st.session_state.last_category,
                        "freshness_score": st.session_state.last_score,
                        "competitor_avg": comp_input
                    }
                    p_res = requests.post(f"{API_URL}/api/calculate-price", json=pricing_payload)
                    if p_res.status_code == 200:
                        p_data = p_res.json()
                        st.success("System Price Calculated Successfully!")
                        
                        col_p1, col_p2, col_p3 = st.columns(3)
                        col_p1.metric("Base Market Price", f"₹{p_data.get('base_market_price')}/kg")
                        col_p2.metric("Recommended System Price", f"₹{p_data.get('recommended_system_price')}/kg", delta=f"Based on {st.session_state.last_score}% Freshness")
                        col_p3.metric("Competitor Benchmark", f"₹{comp_input}/kg")
                        
                        st.info(f"**Strategy Note:** {p_data.get('pricing_strategy')}")
                    else:
                        st.error("Failed to compute system price.")
                except Exception as e:
                    st.error(f"Error connecting to pricing engine: {e}")

    with tab2:
        st.header("Business Inventory & Competitor Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Log Previous Sale")
            s_prod = st.selectbox("Product", ["Tomato", "Apple", "Banana", "Lemon"], key="s_prod")
            s_qty = st.number_input("Quantity Sold (kg)", min_value=0.1, value=10.0)
            s_rev = st.number_input("Total Revenue (₹)", min_value=1.0, value=400.0)
            s_date = st.date_input("Sale Date").strftime("%Y-%m-%d")
            if st.button("Save Sale"):
                try:
                    requests.post(f"{API_URL}/api/sale", json={
                        "username": st.session_state.username, 
                        "product": s_prod, 
                        "quantity_sold": s_qty, 
                        "total_revenue": s_rev, 
                        "sale_date": s_date
                    })
                    st.success("Sale logged!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving sale: {e}")

        with col2:
            st.subheader("Update Stock")
            st_prod = st.selectbox("Product Stock", ["Tomato", "Apple", "Banana", "Lemon"], key="st_prod")
            st_qty = st.number_input("Remaining Stock (kg)", min_value=0.0, value=50.0)
            if st.button("Update Stock"):
                try:
                    requests.post(f"{API_URL}/api/stock", json={
                        "username": st.session_state.username, 
                        "product": st_prod, 
                        "stock_quantity": st_qty
                    })
                    st.success("Stock updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating stock: {e}")

        with col3:
            st.subheader("Add Competitor Price")
            c_prod = st.selectbox("Product Comp", ["Tomato", "Apple", "Banana", "Lemon"], key="c_prod")
            c_name = st.text_input("Competitor Name")
            c_price = st.number_input("Competitor Price (₹/kg)", min_value=1.0, value=40.0)
            if st.button("Save Competitor Price"):
                try:
                    requests.post(f"{API_URL}/api/competitor", json={
                        "username": st.session_state.username, 
                        "product": c_prod, 
                        "competitor_name": c_name, 
                        "price_per_unit": c_price
                    })
                    st.success("Competitor saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving competitor: {e}")

        st.divider()
        st.subheader("📋 Manage Records & Deletions")
        
        try:
            dash_res = requests.get(f"{API_URL}/api/dashboard/{st.session_state.username}")
            if dash_res.status_code == 200:
                d_data = dash_res.json()
                
                m_tab1, m_tab2, m_tab3 = st.tabs(["Manage Sales", "Manage Stock", "Manage Competitors"])
                
                with m_tab1:
                    sales_data = d_data.get("sales", [])
                    if sales_data:
                        df_sales = pd.DataFrame(sales_data)
                        st.dataframe(df_sales, use_container_width=True)
                        sale_id_to_del = st.selectbox("Select Sale ID to Delete", [s["id"] for s in sales_data], key="del_sale")
                        if st.button("Delete Selected Sale"):
                            del_res = requests.delete(f"{API_URL}/api/record", json={"username": st.session_state.username, "record_id": sale_id_to_del, "record_type": "sale"})
                            if del_res.status_code == 200:
                                st.success("Sale record deleted!")
                                st.rerun()
                    else:
                        st.info("No sales recorded yet.")
                        
                with m_tab2:
                    stock_data = d_data.get("stocks", [])
                    if stock_data:
                        st.dataframe(pd.DataFrame(stock_data), use_container_width=True)
                        stock_id_to_del = st.selectbox("Select Stock ID to Delete", [s["id"] for s in stock_data], key="del_stock")
                        if st.button("Delete Selected Stock"):
                            del_res = requests.delete(f"{API_URL}/api/record", json={"username": st.session_state.username, "record_id": stock_id_to_del, "record_type": "stock"})
                            if del_res.status_code == 200:
                                st.success("Stock record deleted!")
                                st.rerun()
                    else:
                        st.info("No stock recorded yet.")
                        
                with m_tab3:
                    comp_data = d_data.get("competitors", [])
                    if comp_data:
                        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
                        comp_id_to_del = st.selectbox("Select Competitor ID to Delete", [c["id"] for c in comp_data], key="del_comp")
                        if st.button("Delete Selected Competitor Price"):
                            del_res = requests.delete(f"{API_URL}/api/record", json={"username": st.session_state.username, "record_id": comp_id_to_del, "record_type": "competitor"})
                            if del_res.status_code == 200:
                                st.success("Competitor record deleted!")
                                st.rerun()
                    else:
                        st.info("No competitor records saved yet.")

                st.divider()
                st.subheader("📈 Product-Wise Pricing & Market Analysis")
                
                user_prices = d_data.get("user_avg_prices", {})
                comp_prices = d_data.get("comp_avg_prices", {})
                
                products = ["Tomato", "Apple", "Banana", "Lemon"]
                my_vals = [user_prices.get(p, 0.0) for p in products]
                comp_vals = [comp_prices.get(p, 0.0) for p in products]
                
                fig, ax = plt.subplots(figsize=(8, 4.5))
                x = np.arange(len(products))
                width = 0.35
                
                rects1 = ax.bar(x - width/2, my_vals, width, label='Your Avg Price', color='#4CAF50')
                rects2 = ax.bar(x + width/2, comp_vals, width, label='Competitor Avg Price', color='#FF9800')
                
                ax.set_ylabel('Price per Unit (₹/kg)')
                ax.set_title('Product-Wise Price Comparison: You vs. Competitors')
                ax.set_xticks(x)
                ax.set_xticklabels(products)
                ax.legend()
                
                ax.bar_label(rects1, padding=3, fmt='₹%.1f')
                ax.bar_label(rects2, padding=3, fmt='₹%.1f')
                
                fig.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.info("Log in and add records to view live dashboard metrics and deletion options.")