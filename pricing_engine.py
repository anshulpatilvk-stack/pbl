def calculate_dynamic_price(base_price: float, competitor_avg: float, freshness_score: float, shelf_life_days: float):
    if freshness_score < 0.20 or shelf_life_days < 0.5:
        return 0.0, "Discard - Zero Hunger Donation"

    floor_price = base_price * 0.50

    if freshness_score >= 0.85:
        system_price = base_price
    elif freshness_score >= 0.50:
        degradation = 1.0 - freshness_score
        discount_factor = degradation * 0.35 
        calculated_price = base_price * (1 - discount_factor)
        system_price = min(calculated_price, competitor_avg * 0.95)
    else:
        degradation = 1.0 - freshness_score
        discount_factor = degradation * 0.60
        system_price = base_price * (1 - discount_factor)

    final_price = max(system_price, floor_price)
    return round(final_price, 2), "Active Sale"