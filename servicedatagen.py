import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import openai
from openai import OpenAI
import time
from io import BytesIO
from zipfile import ZipFile
import base64
import re
# Initialize Faker
fake = Faker()

# Define columns
columns = ['Supplier Name', 'Supplier ID', 'Product Name', 'Product ID', 'Category',
           'Subcategory', 'Description', 'Specification', 'Catalog ID', 'Contract ID',
           'PO Number', 'Unit Price', 'Quantity', 'Avg Cost per Hour', 'Hours Worked',
           'Fixed Cost', 'PO Amount', 'Invoice Amount', 'Currency', 'PO Date',
           'Invoice Date', 'Delivery Date', 'Goods Receipt Date', 'Cost Center',
           'GL Account ID', 'User Name', 'Minority_supplier_certificate',
           'Sustainability_rating', 'Financial_health_score', 'Quality_inspection', 
           'form_id', 'form_description']

# Set your OpenAI API key
client = OpenAI(
    api_key='sk-proj-FKtTIkSIPsXuK3gHbpmFT3BlbkFJBVfclmSCoyZF7PtKbIqo'
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
        model="gpt-4o-mini",
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
        model="gpt-4o",
        max_tokens=60,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content.strip()

def generate_description_suppname(subcategory, total_requested):
    unique_suppliers = total_requested // 5
    repetitions = 5

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Generate supplier names for category {subcategory}. Provide {unique_suppliers} unique supplier names with each name repeated {repetitions} times in a balanced way. Strictly give just the supplier names in your response."
            }
        ],
        model="gpt-4o",
        max_tokens=60,
        temperature=0.2,
    )
    return chat_completion.choices[0].message.content.strip().split('\n')

# Function to clean supplier names by removing numbers and symbols
def clean_supplier_name(name):
    return re.sub(r'[^A-Za-z\s]', '', name)


# Function to generate random data
def generate_random_data(category, subcategory, cost_center, gl_account_id, currency, n):
    data = []
    supplier_ids = {}
    product_ids = {}
    contract_ids = {}

    supplier_names = generate_description_suppname(subcategory, n)
    supplier_index = 0
    for _ in range(n):
        supplier_name = supplier_names[supplier_index]
        supplier_index = (supplier_index + 1) % len(supplier_names)
        
        if supplier_name in supplier_ids:
            supplier_id = supplier_ids[supplier_name]
        else:
            supplier_id = fake.unique.bothify(text='SUP###')
            supplier_ids[supplier_name] = supplier_id
        product_name = generate_description_productname(category, supplier_name)
        if (supplier_name, product_name) in product_ids:
            product_id = product_ids[(supplier_name, product_name)]
        else:
            product_id = fake.unique.bothify(text='PROD###')
            product_ids[(supplier_name, product_name)] = product_id

        description = generate_description(product_name)
        specification = f"Resolution: {np.random.choice(['1200x1200', '2400x1200', '4800x1200'])} dpi, Connectivity: {np.random.choice(['USB', 'Wireless', 'Ethernet'])}"
        catalog_id = np.nan
        if supplier_name in contract_ids:
            contract_id = contract_ids[supplier_name]
        else:
            contract_id = fake.unique.bothify(text='CON####')
            contract_ids[supplier_name] = contract_id

        po_number = fake.unique.bothify(text='PO####')
        quantity = np.nan
        unit_price = np.nan
        avg_cost_per_hour = np.random.uniform(10, 100) if np.random.rand() > 0.5 else np.nan
        hours_worked = np.random.uniform(1, 50) if np.random.rand() > 0.5 else np.nan
        fixed_cost = np.random.uniform(100, 1000) if np.random.rand() <= 0.5 else np.nan
        po_amount = np.nan if fixed_cost else round(unit_price * quantity, 2)
        invoice_amount = po_amount
        po_date = fake.date_between(start_date='-1y', end_date='today')
        delivery_date = po_date + pd.Timedelta(days=np.random.randint(5, 16))
        goods_receipt_date = delivery_date + pd.Timedelta(days=np.random.randint(0, 4))  # Max 3 days after invoice_date
        invoice_date = goods_receipt_date + pd.Timedelta(days=np.random.randint(0, 4))

        form_id = f"F_{category[:3]}-{subcategory[:4]}-{currency}_{supplier_name[:3]}"
        form_description = f"Use this form for {subcategory} related activities."

        user_name = fake.name()
        minority_supplier_certificate = np.random.choice(['yes', 'no'])
        sustainability_rating = round(np.random.uniform(1, 5), 1)  # Float between 1 and 5
        financial_health_score = round(np.random.uniform(1, 5), 1)  # Float between 1 and 5
        quality_inspection = np.random.choice(['pass', 'fail'])

        row = [supplier_name, supplier_id, product_name, product_id, category,
               subcategory, description, specification, catalog_id, contract_id,
               po_number, unit_price, quantity, avg_cost_per_hour, hours_worked,
               fixed_cost, po_amount, invoice_amount, currency, po_date, invoice_date,
               delivery_date, goods_receipt_date, cost_center, gl_account_id, user_name,
               minority_supplier_certificate, sustainability_rating, financial_health_score,
               quality_inspection, form_id, form_description]

        data.append(row)

    return pd.DataFrame(data, columns=columns)

# Streamlit UI
st.set_page_config(
    page_title="CSV Generator",
    page_icon="📈",  # Data emoji
    layout="centered",
    initial_sidebar_state="auto",
)

# Load and encode the image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("D:\\AI_team\\Sql_bot\\Easework logo.png")
emoji_base64 = get_base64_image("emoji.png")

# Add logo and style it
st.markdown(f"""
    <style>
        .logo {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 250px; /* Adjust the width as needed */
            height: auto;
            margin-bottom: 20px; /* Adjust margin-bottom as needed */
            border: 2px solid #5e17eb; /* Border around the logo */
            border-radius: 20px; /* Rounded corners */
            box-shadow: 2px 6px 12px rgba(0, 0, 0, 0.1); /* Shadow effect */
        }}
        .custom-button {{
            background-color: #dfb5f6; /* Button color */
            color: black; /* Text color */
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
            width: 40px; /* Adjust the size as needed */
            height: auto;
        }}
        .input-text {{
            color: #5e17eb; /* colour text */
            font-size: 25px;
            margin-top: 5px; /* Adjust margin-top as needed */
            margin-bottom: 5px; /* Adjust margin-bottom as needed */
        }}
    </style>
    <img src="data:image/png;base64,{logo_base64}" class="logo">
    <div class="title-container">
        <img src="data:image/png;base64,{emoji_base64}" class="emoji" alt="Data Emoji">
        <h1 class="input-text">Service PO Data Generation</h1>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for storing generated data
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = []

with st.form("input_form"):
    cost_center = st.text_input("Cost Center", value='CC83')
    gl_account_id = st.text_input("GL Account ID", value='GL3338')
    subcategory = st.text_input("Subcategory", value='Cyber security')
    currency = st.text_input("Currency", value='USD')
    category = st.text_input("Category", value='IT Consulting Services')
    no_rows = st.number_input("Number of Rows", min_value=1, value=50)
    no_files = st.number_input("Number of Files", min_value=1, value=1)
    submit_button = st.form_submit_button(label="Generate CSV")

if submit_button:
    with st.spinner("⚙️Preparing Data..."):
        for i in range(no_files):
            df = generate_random_data(category, subcategory, cost_center, gl_account_id, currency, no_rows)
            # Clean supplier names
            df['Supplier Name'] = df['Supplier Name'].apply(clean_supplier_name)
            st.session_state.dataframes.append(df)
            st.progress((i + 1) / no_files)

    if 'dataframes' in st.session_state and st.session_state.dataframes:
        if len(st.session_state.dataframes) == 1:
            # Direct download for a single CSV file
            df = st.session_state.dataframes[0]
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="Download CSV File",
                data=csv_buffer,
                file_name=f"{category}_data.csv",
                mime='text/csv',
                key="single_csv_download"
            )
        else:
            # Create a zip file in memory for multiple CSV files
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, 'w') as zf:
                for idx, df in enumerate(st.session_state.dataframes):
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    zf.writestr(f"{category}_data_{idx+1}.csv", csv_buffer.getvalue())
            zip_buffer.seek(0)
            st.download_button(
                label="Download All CSV Files as ZIP",
                data=zip_buffer,
                file_name=f"{category}_data_files.zip",
                mime='application/zip',
                key="zip_download"
            )
