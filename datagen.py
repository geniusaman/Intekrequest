import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from openai import OpenAI
import time
from io import BytesIO
from zipfile import ZipFile
import base64
import re
import os

# Initialize Faker
fake = Faker()

# Define columns
columns = ['Supplier Name', 'Supplier ID', 'Product Name', 'Product ID', 'Category',
           'Subcategory', 'Description', 'Specification', 'Catalog ID', 'Contract ID',
           'PO Number', 'Unit Price', 'Quantity', 'Unit_Of_Measure', 'Avg Cost per Hour',
           'Hours Worked', 'Fixed Cost', 'PO Amount', 'Invoice Amount', 'Currency', 
           'PO Date', 'Invoice Date', 'Delivery Date', 'Goods Receipt Date', 'Department',
           'Cost Center', 'GL Account ID', 'Requestor', 'Minority_supplier_certificate',
           'Sustainability_rating', 'Financial_health_score', 'Quality_inspection']

api_key = st.secrets["openai"]["OPENAI_API_KEY"]
client = OpenAI(
    api_key=api_key
)


# Function to generate product description using GPT-3.5
def generate_description(product_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"generate a description with specification for a product {product_name} in 10-15 words",
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=60,
        temperature=0.7,
    )
    return chat_completion.choices[0].message.content.strip()

def generate_description_productname(subcategory, Supplier_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"every time generate only one product name for a subcategory {subcategory} and Supplier name {Supplier_name}\n *strictly Give just the product name in your response*",
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=60,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content.strip()

def generate_description_suppname(subcategory, total_requested):
    unique_suppliers = int(supplier_id_end) - int(supplier_id_start)
    repetitions = int(total_requested/(15*(total_requested/100)))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Generate supplier names for category {subcategory}. Provide {unique_suppliers} unique supplier names with each name repeated {repetitions} times in a balanced way. Strictly give just the supplier names in your response."
            }
        ],
        model="gpt-3.5-turbo",
        max_tokens=60,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content.strip().split('\n')

# Function to clean supplier names by removing numbers and symbols
def clean_supplier_name(name):
    return re.sub(r'[^A-Za-z\s]', '', name)

# Define the department to cost center mapping
department_cost_center_mapping = {
    'Maintenance': 'CC80',
    'Finance': 'CC81',
    'Warehouse': 'CC82',
    'Marketing': 'CC83',
    'Assembly': 'CC84',
    'Production': 'CC85'
}

# Function to generate random data
def generate_random_data(category, subcategory, gl_account_id, currency, n, supplier_id_start, supplier_id_end):
    data = []
    supplier_ids = {}
    product_ids = {}
    catalog_ids = {}
    supplier_certificates = {}

    supplier_id_range = iter(range(supplier_id_start, supplier_id_end))
    supplier_names = generate_description_suppname(subcategory, n)
    supplier_index = 0
    
    for _ in range(n):
        supplier_name = supplier_names[supplier_index]
        supplier_index = (supplier_index + 1) % len(supplier_names)
        
        cleaned_supplier_name = clean_supplier_name(supplier_name)
        
        if supplier_name in supplier_ids:
            supplier_id = supplier_ids[supplier_name]
        else:
            supplier_id = f"SUP{next(supplier_id_range):03d}"
            supplier_ids[supplier_name] = supplier_id

        product_name = generate_description_productname(subcategory, supplier_name)
        if (supplier_name, product_name) in product_ids:
            product_id = product_ids[(supplier_name, product_name)]
        else:
            product_id = fake.unique.bothify(text='PROD###')
            product_ids[(supplier_name, product_name)] = product_id
     
        description = generate_description(product_name)
        processors = ['Intel i5', 'Intel i7', 'AMD Ryzen 5', 'AMD Ryzen 7']
        ram_options = ['8GB', '16GB', '32GB']
        storage_options = ['256GB SSD', '512GB SSD', '1TB SSD']
        os_options = ['Windows 10 Pro', 'Windows 11 Home', 'Linux Ubuntu']
        if subcategory == "Desktop Computers":
            specification = f"Processor: {np.random.choice(processors)}, RAM: {np.random.choice(ram_options)}, Storage: {np.random.choice(storage_options)}, OS: {np.random.choice(os_options)}"
        else:
            specification = f"Resolution: {np.random.choice(['1200x1200', '2400x1200', '4800x1200'])} dpi, Connectivity: {np.random.choice(['USB', 'Wireless', 'Ethernet'])}"
        if supplier_name in catalog_ids:
            catalog_id = catalog_ids[supplier_name]
        else:
            catalog_id = fake.unique.bothify(text='CAT###')
            catalog_ids[supplier_name] = catalog_id
        #catalog_id = np.nan
        contract_id = np.nan
        po_number = fake.unique.bothify(text='PO####')
        unit_price = round(np.random.uniform(50, 500), 2)
        quantity = np.random.randint(1, 20)
        unit_of_measure = "EA"  # Added Unit_Of_Measure column with value "EA"
        avg_cost_per_hour = np.nan
        hours_worked = np.nan
        fixed_cost = np.nan
        po_amount = round(unit_price * quantity, 2)
        invoice_amount = po_amount
        po_date = fake.date_between(start_date='-1y', end_date='today')
        delivery_date = po_date + pd.Timedelta(days=np.random.randint(5, 16))
        goods_receipt_date = delivery_date + pd.Timedelta(days=np.random.randint(0, 4))
        invoice_date = goods_receipt_date + pd.Timedelta(days=np.random.randint(0, 4))

        user_name = fake.name()
        if cleaned_supplier_name in supplier_certificates:
            minority_supplier_certificate = supplier_certificates[cleaned_supplier_name]
        else:
            minority_supplier_certificate = np.random.choice(['yes', 'no'])
            supplier_certificates[cleaned_supplier_name] = minority_supplier_certificate
        
        sustainability_rating = round(np.random.uniform(1, 5), 1)
        financial_health_score = round(np.random.uniform(1, 5), 1)
        quality_inspection = np.random.choice(['pass', 'fail'])

        department = np.random.choice(list(department_cost_center_mapping.keys()))
        cost_center = department_cost_center_mapping[department]

        row = [supplier_name, supplier_id, product_name, product_id, category,
               subcategory, description, specification, catalog_id, contract_id,
               po_number, unit_price, quantity, unit_of_measure, avg_cost_per_hour, 
               hours_worked, fixed_cost, po_amount, invoice_amount, currency, po_date, 
               invoice_date, delivery_date, goods_receipt_date, department, cost_center, 
               gl_account_id, user_name, minority_supplier_certificate, 
               sustainability_rating, financial_health_score, quality_inspection]

        data.append(row)

    return pd.DataFrame(data, columns=columns)

# Streamlit UI
st.set_page_config(
    page_title="CSV Generator",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)

# Load and encode the image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("Easework logo.png")
emoji_base64 = get_base64_image("emoji.png") 

# Add logo and style it
st.markdown(f"""
    <style>
        .logo {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 250px;
            height: auto;
            margin-bottom: 20px;
            border: 2px solid #5e17eb;
            border-radius: 20px;
            box-shadow: 2px 6px 12px rgba(0, 0, 0, 0.1);
        }}
        .custom-button {{
            background-color: #dfb5f6;
            color: black;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }}
        .title-container {{
            display: flex;
            align-items: center;
            justify-content: left;
            gap: 10px;
        }}
        .emoji {{
            vertical-align: middle;
            width: 40px;
            height: auto;
        }}
        .input-text {{
            color: #5e17eb;
            font-size: 25px;
            margin-top: 5px;
            margin-bottom: 5px;
        }}
    </style>
    <img src="data:image/png;base64,{logo_base64}" class="logo">
    <div class="title-container">
        <img src="data:image/png;base64,{emoji_base64}" class="emoji" alt="Data Emoji">
        <h1 class="input-text">Goods PO Data Generation</h1>
    </div>
""", unsafe_allow_html=True)

_state = st.session_state
if "dataframes" not in _state:
    _state.dataframes = []

# Streamlit form
with st.form("data_generation_form"):
    st.subheader("Enter your details:")
    category = st.text_input('Category:')
    subcategory = st.text_input('Subcategory:')
    gl_account_id = st.text_input('GL Account ID:')
    currency = st.selectbox('Currency:', ['USD', 'EUR', 'GBP', 'INR'], index=0)
    supplier_id_start = st.number_input("Starting Supplier ID Range:", min_value=0, value=0)
    supplier_id_end = st.number_input("Ending Supplier ID Range:", min_value=1, value=100)
    no_rows = st.number_input("Number of Rows", min_value=1, value=10)
    no_files = st.number_input("Number of Files", min_value=1, value=1)
    submit_button = st.form_submit_button(label="Generate CSV")

def calculate_quality_score(df):
    # Encode 'Quality_inspection' column: 'pass' -> 1, 'fail' -> 0
    df['Quality_inspection_encoded'] = df['Quality_inspection'].apply(lambda x: 1 if x == 'pass' else 0)
    
    # Group by 'Supplier Name' and calculate the average of 'Quality_inspection_encoded'
    quality_scores = df.groupby('Supplier Name')['Quality_inspection_encoded'].mean()
    
    # Multiply the average quality score by 5
    quality_scores *= 5
    
    # Merge the quality scores back into the original DataFrame
    df = df.merge(quality_scores.reset_index(name='Quality_Score'), on='Supplier Name', how='left')
    
    # Drop the temporary encoded column
    df.drop(columns=['Quality_inspection_encoded'], inplace=True)
    
    return df

def calculate_ontime_delivery_score(df):
    # Ensure date columns are in datetime format
    df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])
    df['Goods Receipt Date'] = pd.to_datetime(df['Goods Receipt Date'])
    
    # Calculate the difference in days between Delivery Date and Goods Receipt Date
    df['Days_Difference'] = (df['Goods Receipt Date'] - df['Delivery Date']).dt.days
    
    # Group by 'Supplier Name' and calculate the average of 'Days_Difference'
    avg_days_difference = df.groupby('Supplier Name')['Days_Difference'].mean()
    
    # Calculate the On-Time Delivery Score by subtracting the average difference from 5
    ontime_delivery_score = 5 - avg_days_difference
    
    # Merge the On-Time Delivery Score back into the original DataFrame
    df = df.merge(ontime_delivery_score.reset_index(name='On_Time_Delivery_Score'), on='Supplier Name', how='left')
    
    # Drop the temporary column
    df.drop(columns=['Days_Difference'], inplace=True)
    
    return df

# Example usage within your existing Streamlit application
if submit_button:
    with st.spinner("⚙️Preparing Data..."):
        st.session_state.dataframes.clear()
        
        for i in range(no_files):
            df = generate_random_data(category, subcategory, gl_account_id, currency, no_rows, supplier_id_start, supplier_id_end)
            df['Supplier Name'] = df['Supplier Name'].apply(clean_supplier_name)
            # Calculate and include the quality score
            df = calculate_quality_score(df)
            # Calculate and include the on-time delivery score
            df = calculate_ontime_delivery_score(df)
            st.session_state.dataframes.append(df)
            st.progress((i + 1) / no_files)

    if 'dataframes' in st.session_state and st.session_state.dataframes:
        if len(st.session_state.dataframes) == 1:
            df = st.session_state.dataframes[0]
            # Save the updated DataFrame to CSV
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.dataframe(df)
            st.download_button(
                label="Download CSV File",
                data=csv_buffer,
                file_name=f"{category}_data_with_scores.csv",
                mime='text/csv',
                key="single_csv_download"
            )
        else:
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, 'w') as zf:
                for idx, df in enumerate(st.session_state.dataframes):
                    # Calculate and include the quality score and on-time delivery score
                    df = calculate_quality_score(df)
                    df = calculate_ontime_delivery_score(df)
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    zf.writestr(f"{category}_data_with_scores_{idx+1}.csv", csv_buffer.getvalue())
                    
            zip_buffer.seek(0)
            st.download_button(
                label="Download All CSV Files as ZIP",
                data=zip_buffer,
                file_name=f"{category}_data_files_with_scores.zip",
                mime='application/zip',
                key="zip_download"
            )
