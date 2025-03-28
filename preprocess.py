import pandas as pd
from datetime import datetime

def process_sales_data(data):
    # Convert date columns to datetime
    data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
    data['Date Shipped'] = pd.to_datetime(data['Date Shipped'], errors='coerce')
    data['Due Date'] = pd.to_datetime(data['Due Date'], errors='coerce')
    
    # Handle missing values
    data.fillna(0, inplace=True)
    
    # Calculate order delays
    data['Shipping Delay'] = (data['Date Shipped'] - data['Due Date']).dt.days
    data['Shipped Late'] = data['Shipping Delay'] > 0
    
    # Calculate profit margin
    data['Profit Margin'] = data['Total Price'] - (data['Quantity Ordered'] * data['Unit Price'])
    
    # Filter data based on Customer ID
    customer_ids = data['Customer ID'].unique()
    filtered_data = data[data['Customer ID'].isin(customer_ids)]
    
    # 1. Monthly Sales Totals
    monthly_totals = filtered_data.groupby(filtered_data['Order Date'].dt.to_period('M'))['Total Price'].sum()
    
    # 2. Year-to-Date Sales
    current_year = datetime.now().year
    ytd_totals = filtered_data[filtered_data['Order Date'].dt.year == current_year].groupby(filtered_data['Order Date'].dt.to_period('M'))['Total Price'].sum()
    
    # 3. Average Order Value based on user query
    average_order_values = {}
    for period, group in filtered_data.groupby(filtered_data['Order Date'].dt.to_period('M')):
        total_sales = group['Total Price'].sum()
        number_of_orders = group['Quantity Ordered'].sum()
        
        if number_of_orders > 0:
            average_order_values[period] = total_sales / number_of_orders
        else:
            average_order_values[period] = 0  # or handle as needed
    
    # 4. Top 5 Customers by Total Spending
    top_customers_by_spending = filtered_data.groupby('Customer ID')['Total Price'].sum().nlargest(5)
    
    # 5. Return Rates by Product
    return_rates = filtered_data.groupby('Item ID').agg({'Quantity  Returned': 'sum', 'Quantity Shipped': 'sum'})
    return_rates['Return Rate'] = return_rates['Quantity  Returned'] / return_rates['Quantity Shipped']
    return_rates = return_rates.sort_values(by='Return Rate', ascending=False)
    
    # 6. Shipments by Warehouse and Ship Code
    shipments_by_warehouse = filtered_data.groupby('Ship Warehouse')['Quantity Shipped'].sum().sort_values(ascending=False)
    shipments_by_code = filtered_data['Ship Code'].value_counts()
    
    # 7. Customer Retention (Customers who placed repeat orders)
    customer_order_counts = filtered_data.groupby('Customer ID')['Order ID'].nunique()
    repeat_customers = customer_order_counts[customer_order_counts > 1]
    retention_rate = len(repeat_customers) / len(customer_order_counts)
    
    # 8. Late Shipments Count
    late_shipments_count = filtered_data['Shipped Late'].sum()
    
    # 9. Most Frequent Unit of Measure
    most_frequent_uom = filtered_data[' Unit of Measure'].mode()[0]
    
    # Prepare context
    contexts = [
        f"Monthly Sales Totals:\n{monthly_totals.to_string()}",
        f"Year-to-Date Sales Totals:\n{ytd_totals.to_string()}",
        f"Average Order Values:\n" + "\n".join([f"{period}: ${average_order_values[period]:.2f}" for period in average_order_values]),
        f"Top 5 Customers by Total Spending:\n{top_customers_by_spending.to_string()}",
        f"Return Rates by Product:\n{return_rates.to_string()}",
        f"Shipments by Warehouse:\n{shipments_by_warehouse.to_string()}",
        f"Shipments by Ship Code:\n{shipments_by_code.to_string()}",
        f"Customer Retention Rate: {retention_rate:.2%}",
        f"Total Late Shipments: {late_shipments_count}",
        f"Most Frequent Unit of Measure: {most_frequent_uom}",
    ]
    
    return contexts